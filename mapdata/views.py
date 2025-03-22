from django.views import View
from django.http import JsonResponse
import json
from django.middleware.csrf import get_token
import geopandas as gpd
from .calculator import wgs1984_to_cgcs, demPointsToArray, smooth_filter, arrayToDemPoints, cal_earth_work
import numpy as np
from rasterio import features, transform
from scipy.interpolate import griddata
from shapely.geometry import Point
import pandas as pd


def UploadData(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '非POST请求方法，无效！'})

    try:
        # (1) 获取请求体中的GeoJSON数据
        get_data = json.loads(request.body)
        dem_data = get_data["demData"]
        area_data = get_data["areaData"]
        col_name = get_data["colName"]
        resolution = float(get_data["resolution"])

        # (2) 验证GeoJSON格式（示例验证）
        if dem_data.get('type') != 'FeatureCollection':
            return JsonResponse({'error': '无效的GeoJSON格式'}, status=400)
        if area_data.get('type') != 'FeatureCollection':
            return JsonResponse({'error': '无效的GeoJSON格式'}, status=400)

        # (3) 处理数据
        dem_gdf = gpd.GeoDataFrame.from_features(dem_data['features'], crs=dem_data['crs']['properties']['name'])
        dem_gdf = wgs1984_to_cgcs(dem_gdf.to_crs(4326))
        area_gdf = gpd.GeoDataFrame.from_features(area_data['features'], crs=area_data['crs']['properties']['name'])
        area_gdf = wgs1984_to_cgcs(area_gdf.to_crs(4326))

        minx, miny, maxx, maxy = dem_gdf.total_bounds
        # 把点转为栅格，边界需要外扩半个栅格像素
        minx = minx - resolution / 2
        miny = miny - resolution / 2
        maxx = maxx + resolution / 2
        maxy = maxy + resolution / 2
        # 点转栅格
        rasterized = demPointsToArray(dem_gdf, resolution, col_name)

        # 对于未赋值的像素，使用插值方法填补
        points = np.array([(pt.x, pt.y) for pt in dem_gdf.geometry])
        values = dem_gdf[col_name].values

        # 创建整个栅格的坐标网格
        x_coords = np.linspace(minx, maxx, rasterized.shape[1])
        y_coords = np.linspace(maxy, miny, rasterized.shape[0])  # 注意这里的顺序，因为raster是从上到下递增
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)

        # 使用插值填充缺失值
        filled_raster = griddata(points=points, values=values, xi=(x_mesh, y_mesh), method='cubic')

        # 构建点gdf
        dem_gdf = arrayToDemPoints(filled_raster, resolution, col_name, minx, maxy, dem_gdf.crs)
        dem_gdf = dem_gdf.loc[dem_gdf["geometry"].within(area_gdf.unary_union), :]
        dem_gdf = dem_gdf.loc[dem_gdf[col_name] > 0, :]

        # 转为84坐标后转json，同时手动写入坐标系
        dem_geosjon = dem_gdf.to_crs(4326).to_json()
        dem_geosjon = json.loads(dem_geosjon)
        dem_geosjon['crs'] = {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        }
        dem_geosjon = json.dumps(dem_geosjon)

        return JsonResponse({
            'success': True,
            'orinDem': dem_geosjon,
            'message': 'GeoJSON数据处理完成！'
        }, status=200)

    except json.JSONDecodeError:
        return JsonResponse({'error': '无效的JSON格式'}, status=400)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def CalSmoothData(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '非POST请求方法，无效！'})
    try:
        # (1) 获取请求体中的GeoJSON数据
        get_data = json.loads(request.body)
        dem_data = get_data["demData"]
        col_name = get_data["colName"]
        sigma = float(get_data["sigma"])
        n_cluster = int(get_data["nCluster"])
        truncate = float(get_data["truncate"])
        resolution = float(get_data["resolution"])

        # (2) 验证GeoJSON格式（示例验证）
        if dem_data.get('type') != 'FeatureCollection':
            return JsonResponse({'error': '无效的GeoJSON格式'}, status=400)

        # (3) 处理数据
        dem_gdf = gpd.GeoDataFrame.from_features(dem_data['features'], crs=dem_data['crs']['properties']['name'])
        dem_gdf = wgs1984_to_cgcs(dem_gdf.to_crs(4326))

        # (4) 点转栅格
        rasterized = demPointsToArray(dem_gdf, resolution, col_name)

        # (5) 开始计算
        smoothed_raster, labels_array = smooth_filter(rasterized, sigma, n_cluster, truncate)

        # (7) 还原为gdf
        smoothed_gdf = arrayToDemPoints(smoothed_raster, resolution, col_name, dem_gdf.total_bounds[0],
                                  dem_gdf.total_bounds[3], dem_gdf.crs, labels_array=labels_array)

        # 计算土方量
        smoothed_gdf = cal_earth_work(dem_gdf, smoothed_gdf, col_name, resolution)

        # 转为84坐标后转json，同时手动写入坐标系
        smoothed_geosjon = smoothed_gdf.to_crs(4326).to_json()
        smoothed_geosjon = json.loads(smoothed_geosjon)
        smoothed_geosjon['crs'] = {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        }
        smoothed_geosjon = json.dumps(smoothed_geosjon)

        return JsonResponse({
            'success': True,
            'smoothData': smoothed_geosjon,
            'message': 'GeoJSON数据处理完成！'
        })
    except json.JSONDecodeError:
        return JsonResponse({'error': '无效的JSON格式'}, status=400)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def CalSmoothDataPart(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '非POST请求方法，无效！'})
    try:
        # (1) 获取请求体中的GeoJSON数据
        get_data = json.loads(request.body)
        dem_data = get_data["demData"]
        smooth_data = get_data["smoothData"]
        selected_labels = [int(i) for i in get_data["labels"]]
        col_name = get_data["colName"]
        sigma = float(get_data["sigma"])
        truncate = float(get_data["truncate"])
        resolution = float(get_data["resolution"])

        # (2) 验证GeoJSON格式（示例验证）
        if smooth_data.get('type') != 'FeatureCollection':
            return JsonResponse({'error': '无效的GeoJSON格式'}, status=400)

        # (3) 处理数据
        dem_data_gdf = gpd.GeoDataFrame.from_features(dem_data['features'], crs=dem_data['crs']['properties']['name'])
        dem_data_gdf = wgs1984_to_cgcs(dem_data_gdf.to_crs(4326))
        smooth_data_gdf = gpd.GeoDataFrame.from_features(smooth_data['features'], crs=smooth_data['crs']['properties']['name'])
        smooth_data_gdf = wgs1984_to_cgcs(smooth_data_gdf.to_crs(4326))

        dem_data_gdf_labeled = dem_data_gdf.sjoin_nearest(
            smooth_data_gdf,
            how="left",
        )
        for col in dem_data_gdf_labeled.columns:
            if col[-5:] == '_left':
                dem_data_gdf_labeled.rename(columns={col: col[:-5]}, inplace=True)
        dem_data_gdf_labeled = dem_data_gdf_labeled.loc[:, [col for col in dem_data_gdf.columns] + ["label"]]
        dem_data_gdf_labeled.loc[:, "label"] = dem_data_gdf_labeled.loc[:, "label"].fillna(-1)

        # (4) 点转栅格
        rasterized = demPointsToArray(dem_data_gdf_labeled, resolution, col_name)  # 原始地形栅格
        labeled_label = demPointsToArray(dem_data_gdf_labeled, resolution, "label", add=False)  # 原始地形栅格带分类标签

        # (5) 开始计算
        smoothed_raster, _ = smooth_filter(rasterized, sigma, 0, truncate, labels=np.where(
            np.isin(labeled_label, selected_labels),
            labeled_label,
            np.nan
        ))  # 仅对指定标签范围内的原始地形栅格进行平滑
        smoothed_raster = np.where(
            np.isin(labeled_label, selected_labels),
            smoothed_raster,
            np.nan
        )  # 仅提取出指定标签范围内平滑后的结果

        # (6) 还原为gdf
        smoothed_raster_gdf = arrayToDemPoints(smoothed_raster, resolution, col_name,
                                               dem_data_gdf_labeled.total_bounds[0],
                                               dem_data_gdf_labeled.total_bounds[3],
                                               dem_data_gdf_labeled.crs,
                                               labels_array=np.where(
                                                   np.isin(labeled_label, selected_labels),
                                                   labeled_label,
                                                   np.nan
                                               ))

        # (7) 和其他部分合并
        smoothed_gdf = pd.concat([
            smoothed_raster_gdf,
            smooth_data_gdf.loc[~smooth_data_gdf["label"].isin(selected_labels), :]
        ])

        # (8) 再执行一次转栅格、转gdf的流程，避免新点和旧点之间的间距有问题
        smoothed_array = demPointsToArray(smoothed_gdf, resolution, col_name)
        smoothed_array_label = demPointsToArray(smoothed_gdf, resolution, "label", add=False)
        smoothed_gdf = arrayToDemPoints(smoothed_array, resolution, col_name,
                                        smoothed_gdf.total_bounds[0],
                                        smoothed_gdf.total_bounds[3],
                                        smoothed_gdf.crs,
                                        labels_array=smoothed_array_label)

        # 计算土方量
        smoothed_gdf = cal_earth_work(dem_data_gdf, smoothed_gdf, col_name, resolution)

        # 转为84坐标后转json，同时手动写入坐标系
        smoothed_geosjon = smoothed_gdf.to_crs(4326).to_json()
        smoothed_geosjon = json.loads(smoothed_geosjon)
        smoothed_geosjon['crs'] = {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        }
        smoothed_geosjon = json.dumps(smoothed_geosjon)

        return JsonResponse({
            'success': True,
            'smoothData': smoothed_geosjon,
            'message': 'GeoJSON数据处理完成！'
        })
    except json.JSONDecodeError:
        return JsonResponse({'error': '无效的JSON格式'}, status=400)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse({'error': str(e)}, status=500)


def CalSmoothDataFinal(request):
    if request.method != 'POST':
        return JsonResponse({'success': False, 'message': '非POST请求方法，无效！'})
    try:
        # (1) 获取请求体中的GeoJSON数据
        get_data = json.loads(request.body)
        dem_data = get_data["demData"]
        smooth_data = get_data["smoothDataFinal"]
        idx = [int(i) for i in get_data["idx"]]
        col_name = get_data["colName"]
        sigma = float(get_data["sigma"])
        truncate = float(get_data["truncate"])
        resolution = float(get_data["resolution"])

        # (2) 验证GeoJSON格式（示例验证）
        if smooth_data.get('type') != 'FeatureCollection':
            return JsonResponse({'error': '无效的GeoJSON格式'}, status=400)

        # (3) 处理数据
        smooth_data_gdf = gpd.GeoDataFrame.from_features(smooth_data['features'], crs=smooth_data['crs']['properties']['name'])
        smooth_data_gdf = wgs1984_to_cgcs(smooth_data_gdf.to_crs(4326))
        dem_data_gdf = gpd.GeoDataFrame.from_features(dem_data['features'], crs=dem_data['crs']['properties']['name'])
        dem_data_gdf = wgs1984_to_cgcs(dem_data_gdf.to_crs(4326))
        
        # (4) 点转栅格
        rasterized = demPointsToArray(smooth_data_gdf, resolution, col_name)
        rasterized_idx = demPointsToArray(smooth_data_gdf, resolution, "idx", add=False)
        rasterized_label = demPointsToArray(smooth_data_gdf, resolution, "label", add=False)
        
        # (5) 对整个栅格进行平滑，但只对指定标签范围内的原始栅格进行取值
        smoothed_raster, _ = smooth_filter(rasterized, sigma, 1, truncate)
        smoothed_raster = np.where(
            np.isin(rasterized_idx, idx),
            smoothed_raster,
            rasterized
        )

        # (6) 还原为gdf
        smoothed_raster_gdf = arrayToDemPoints(smoothed_raster, resolution, col_name,
                                               smooth_data_gdf.total_bounds[0],
                                               smooth_data_gdf.total_bounds[3],
                                               smooth_data_gdf.crs,
                                               labels_array=rasterized_label
                                               )

        # 计算土方
        smoothed_raster_gdf = cal_earth_work(dem_data_gdf, smoothed_raster_gdf, col_name, resolution)

        # 转为84坐标后转json，同时手动写入坐标系
        smoothed_geosjon = smoothed_raster_gdf.to_crs(4326).to_json()
        smoothed_geosjon = json.loads(smoothed_geosjon)
        smoothed_geosjon['crs'] = {
            "type": "name",
            "properties": {
                "name": "EPSG:4326"
            }
        }
        smoothed_geosjon = json.dumps(smoothed_geosjon)

        return JsonResponse({
            'success': True,
            'smoothDataFinal': smoothed_geosjon,
            'message': 'GeoJSON数据处理完成！'
        })
    except json.JSONDecodeError:
        return JsonResponse({'error': '无效的JSON格式'}, status=400)
    except Exception as e:
        print(f"An error occurred: {e}")
        return JsonResponse({'error': str(e)}, status=500)