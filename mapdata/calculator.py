import geopandas as gpd
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import label, binary_erosion, generate_binary_structure
from sklearn.mixture import GaussianMixture
from rasterio import features, transform
from rasterio.enums import MergeAlg
from shapely.geometry import Point


def arrayToDemPoints(dem_array, resolution, col_name, minx, maxy, crs, labels_array=None):
    # 在插值完成后，开始生成GeoDataFrame
    geoms = []
    values = []
    labels = []
    # 遍历每个栅格单元
    for row in range(dem_array.shape[0]):
        for col in range(dem_array.shape[1]):
            # 计算当前像元的中点坐标（注意行列索引方向）
            x = minx + (col + 0.5) * resolution
            y = maxy - (row + 0.5) * resolution

            geo = Point(x, y)
            val = dem_array[row, col]
            if labels_array is not None:
                label = labels_array[row, col]

            if ~np.isnan(val):
                geoms.append(geo)
                values.append(val)
                if labels_array is not None:
                    labels.append(label)

    if labels_array is not None:
        # 将结果转换为DataFrame
        dem_gdf = gpd.GeoDataFrame({
            'geometry': geoms,
            'label': labels,
            col_name: values  # 使用用户指定的列名
        }, crs=crs, geometry='geometry')
    else:
        # 将结果转换为DataFrame
        dem_gdf = gpd.GeoDataFrame({
            'geometry': geoms,
            col_name: values  # 使用用户指定的列名
        }, crs=crs, geometry='geometry')

    dem_gdf["idx"] = dem_gdf.index

    return dem_gdf


def demPointsToArray(dem_gdf, resolution, col_name, add=True):
    """
    gdf格式的dem点转成数组
    :param dem_gdf:
    :param resolution:
    :param col_name:
    :return:
    """
    minx, miny, maxx, maxy = dem_gdf.total_bounds
    # 把点转为栅格，边界需要外扩半个栅格像素
    minx = minx - resolution / 2
    miny = miny - resolution / 2
    maxx = maxx + resolution / 2
    maxy = maxy + resolution / 2

    # 创建空的栅格图像
    cols = int((maxx - minx) / resolution)
    rows = int((maxy - miny) / resolution)

    # 将GeoDataFrame中的点转为与栅格对应的坐标和值
    shapes_values = ((geom, float(value)) for geom, value in zip(dem_gdf.geometry, dem_gdf[col_name]))
    shapes_count = ((geom, 1) for geom, value in zip(dem_gdf.geometry, dem_gdf[col_name]))
    # 栅格化

    if add:
        # 计算累加值
        rasterized_add = features.rasterize(shapes=shapes_values,
                                            out_shape=(rows, cols),
                                            fill=0,  # 初始所有没数据的地方都填充0
                                            transform=transform.from_bounds(minx, miny, maxx, maxy, cols,
                                                                            rows),
                                            all_touched=True,
                                            dtype=np.float64,
                                            merge_alg=MergeAlg.add  # 累加
        )
        # 计算数量
        rasterized_count = features.rasterize(shapes=shapes_count,
                                             out_shape=(rows, cols),
                                             fill=0,  # 初始所有没数据的地方都填充0
                                             transform=transform.from_bounds(minx, miny, maxx, maxy, cols,
                                                                             rows),
                                             all_touched=True,
                                             dtype=np.int64,
                                             merge_alg=MergeAlg.add  # 累加
        )

        # 计算均值
        rasterized = np.divide(
            rasterized_add,
            rasterized_count,
            out=np.full_like(rasterized_add, np.nan),  # 默认填充为 NaN
            where=(rasterized_count != 0) & (~np.isnan(rasterized_count)) & (~np.isnan(rasterized_add)),
        )
        # 没有点落在的位置改回NaN
        rasterized = np.where(rasterized_count == 0, np.nan, rasterized)

    else:
        # 计算覆盖值
        rasterized = features.rasterize(shapes=shapes_values,
                                        out_shape=(rows, cols),
                                        fill=np.nan,  # 初始所有没数据的地方都填充nan
                                        transform=transform.from_bounds(minx, miny, maxx, maxy, cols,
                                                                        rows),
                                        all_touched=True,
                                        dtype=np.float64,
        )

    return rasterized


def wgs1984_to_cgcs(geodata):
    """
    根据1984的经纬度，选择合适的CGCS2000投影带进行投影
    :param geodata:
    :return:
    """
    # CGCS2000中央经度带范围75~135，每3度一个带，EPSG范围4534~4554
    # 获取geodata的中央经度
    try:
        center_x = geodata["geometry"].unary_union.centroid.x
    except:
        center_x = geodata["geometry"].centroid.x
    # 坐标转换
    EPSG = int(round((center_x - 75) / 3, 0) + 4534)
    geodata_new = geodata.to_crs(EPSG)

    return geodata_new


def smooth_filter(z, sigma, n_cluster, truncate, labels=None):
    """
    改进的高斯滤波函数（支持NaN值处理）

    参数:
    z (np.ndarray): 输入数据（可能包含NaN）
    sigma (float): 高斯核标准差
    n_cluster (int): 聚类数量，当labels不为None时，该参数无意义
    truncate (float): 核截断范围（单位：sigma）

    返回:
    np.ndarray: 平滑后的数组，NaN区域保持原值
    """
    # 生成有效数据掩码
    mask = np.isnan(z)
    valid = ~mask

    # 创建临时数组（NaN替换为0）
    z_filled = np.where(valid, z, 0)

    # 生成坐标网格
    rows, cols = np.indices(z_filled.shape)
    # 构建特征矩阵（标准化处理）
    features = np.stack([
        rows[valid].ravel() / z_filled.shape[0],  # 归一化行坐标
        cols[valid].ravel() / z_filled.shape[1],  # 归一化列坐标
        z_filled[valid].ravel() / z_filled.max()  # 归一化高程
    ], axis=1)

    if labels is None:
        # 执行高斯混合聚类
        gmm = GaussianMixture(
            n_components=n_cluster,
        ).fit(features)

        probs = gmm.predict_proba(features)

        # 重建标签矩阵
        labels = np.full(z_filled.shape, -1, dtype=int)  # -1表示无效区
        labels[valid] = np.argmax(probs, axis=1)

        # 依据连通性重建分类标签
        for cluster_id in range(0, n_cluster):
            class_mask = (labels == cluster_id) & valid
            # 进行连通域标记（使用8邻域连通）
            labeled_array, num_features = label(class_mask)
            labels = np.where(labeled_array > 0,
                              labeled_array + labels * 1000,
                              labels)

    # 计算高斯权重核
    radius = int(truncate * sigma)
    x, y = np.arange(-radius, radius + 1), np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(x, y)  # 形成方形网格
    kernel = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化

    # 每个簇分别进行高斯滤波
    smoothed = np.zeros_like(z_filled)
    for cluster_id in np.unique(labels[labels != -1]):
        z_filled_label = np.zeros_like(z_filled)
        z_filled_label[labels == cluster_id] = z_filled[labels == cluster_id]

        # 先正常卷积，每处的位置都是 周边值的加权和 / 卷积核的总权重
        z_filled_label_smoothed = fftconvolve(z_filled_label, kernel, mode='same')

        # 做一个非0部分全为1的数值进行卷积，每处的值都是 周边非0位置的权重和 / 卷积核的总权重
        z_filled_label_valid = np.zeros_like(z_filled_label)
        z_filled_label_valid[labels == cluster_id] = 1
        z_filled_label_smoothed_count = fftconvolve(z_filled_label_valid, kernel, mode='same')

        # 二者相除，得到的结果每处的数值就是 周边值的加权和 / 周边非0位置的权重和 实现了非0位置的加权平均
        z_filled_label_smoothed = np.divide(
            z_filled_label_smoothed,
            z_filled_label_smoothed_count,
            where=z_filled_label_smoothed_count > 0
        )

        # 赋值
        smoothed[labels == cluster_id] = z_filled_label_smoothed[labels == cluster_id]

    # 恢复原始NaN区域
    smoothed[mask] = np.nan
    return smoothed, labels


def slope_cal(dem_array, cell_length):
    slop = np.zeros_like(dem_array)
    for i in range(1, dem_array.shape[0] - 1):
        for j in range(1, dem_array.shape[1] - 1):
            # 如果这个点东西南北五个位置的高程存在空值，则无法计算坡度，返回np.nan
            if any([
                np.isnan(dem_array[i, j]),
                np.isnan(dem_array[i, j - 1]), np.isnan(dem_array[i, j + 1]),
                np.isnan(dem_array[i - 1, j]), np.isnan(dem_array[i + 1, j]),
            ]):
                slop[i, j] = np.nan

            else:
                # 东西方向倾斜向量
                coefficients = np.polyfit(
                    [j - cell_length, j, j + cell_length],
                    [
                        dem_array[i, j - 1],
                        dem_array[i, j],
                        dem_array[i, j + 1],
                    ],
                    1
                )
                slope_tan_we_vector = np.array([0, 1, coefficients[0]])
                # 南北方向倾斜向量
                coefficients = np.polyfit(
                    [i - cell_length, i, i + cell_length],
                    [
                        dem_array[i - 1, j],
                        dem_array[i, j],
                        dem_array[i + 1, j],
                    ],
                    1
                )
                slope_tan_ns_vector = np.array([1, 0, coefficients[0]])
                # 计算二者的叉积（法向量）
                c = np.cross(slope_tan_we_vector, slope_tan_ns_vector)
                # 计算法向量与z轴的夹角，即坡度，以度为单位
                slope = np.degrees(
                    np.arccos(
                        abs(np.dot(c, np.array([0, 0, 1])) / (np.linalg.norm(c) * np.linalg.norm(np.array([0, 0, 1]))))
                    )
                )
                slop[i, j] = slope

    return slop


def cal_earth_work(dem_gdf, soomth_gdf, col_name, resolution):
    """
    用于计算每个点的挖填方量
    """
    soomth_gdf_sjoin = gpd.sjoin_nearest(
        soomth_gdf,
        dem_gdf.loc[:, [col_name, "geometry"]],
        how="left",
        lsuffix="left",
        rsuffix="right",
    )
    for col in soomth_gdf_sjoin.columns:
        if col[-5:] == '_left':
            soomth_gdf_sjoin.rename(columns={col: col[:-5]}, inplace=True)

    soomth_gdf_sjoin.drop_duplicates(
        subset=["idx"],
        keep="first",
        inplace=True
    )

    for idx in soomth_gdf_sjoin.index:
        now = soomth_gdf_sjoin.loc[idx, col_name]
        orin = soomth_gdf_sjoin.loc[idx, col_name + "_right"]
        soomth_gdf_sjoin.loc[idx, "earthWork"] = (now - orin) * (resolution ** 2)

    return soomth_gdf_sjoin.loc[:, [col for col in soomth_gdf.columns] + ["earthWork"]]