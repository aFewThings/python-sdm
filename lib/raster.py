"""
"
"   Author: Maximilien Servajean - mservajean
"   Mail: servajean@lirmm.fr
"   Date: 04/01/2019
"
"   Description: The code to extract environmental tensors and environmental vectors given some environmental rasters.
"
"""
import os
import warnings

import numpy as np
import rasterio
import geopandas as gpd
from shapely.geometry import Point
from natsort import natsorted


MIN_ALLOWED_VALUE = -10000
EPS = 1


class Raster(object):
    """
    Raster is dedicated to a single raster management...
    """
    def __init__(self, path, nan=None, normalized=False, transform=None, size=64, one_hot=False,
                 unique_values=None, attrib_column='QUANTI', max_val=255, min_val=0, mean=0, std=1, dtype=None):
        """
        Loads a tiff file describing an environmental raster into a numpy array and...

        :param path: the path of the raster (the directory)
        :param nan: the value to use when NaN number are present. If False, then default values will be used
        :param normalized: if True the raster will be normalized (minus the mean and divided by std)
        :param transform: if a function is given, it will be applied on each patch.
        :param size: the size of a patch (size x size)
        :param one_hot: if True, each patch will have a one hot encoding representation given the values in the raster.
        :param attrib_column: the name of the column that contains the correct value to use in the raster
        :param max_val: the maximum value within the raster (used to reconstruct correct values in the raster)
        :param min_val: the minimum value within the raster (used to reconstruct correct values in the raster)
        """
        self.path = path
        self.normalized = normalized
        self.transform = transform
        self.size = size
        self.one_hot = one_hot

        path = path.replace('//', '/')

        self.name = path.split('/')[-1] if path[-1] != '/' else path.split('/')[-2]

        src = rasterio.open(path + '/' + self.name + '.tif')

        assert src.meta['crs'] is not None, 'Tiff must contain CRS'
        assert src.count == 1, 'Tiff must contain a single band'

        # loading the raster
        self.raster = np.squeeze(src.read())
        self.geospaital_transform = src.transform
        nodata = src.nodata # NOTE: 잘못된 nodata값이 저장된 경우가 있음 (globcover는 nodata가 230이지만, 0.0으로 기록됨)
        src.close()

        if dtype != np.ubyte:
            self.raster = self.raster.astype(np.float32)

        # value bellow min_value are considered incorrect and therefore no_data
        self.nodata_cell = (self.raster == nodata) | (self.raster < MIN_ALLOWED_VALUE) | np.isnan(self.raster)
        if nan is not None:
            self.nodata_cell = self.nodata_cell | (self.raster == nan)

        # get mean, std, nan values within raster directly
        self.mean = np.float32(self.raster[self.nodata_cell != True].mean()) if mean == 0 else mean
        self.std = np.float32(self.raster[self.nodata_cell != True].std()) if std == 1 else std
        #? no_data로 최소값보다 작은 값을 사용중인데, proximity distance에선 최소 값이 0임. 이슈 존재할 수 있음.
        self.no_data = np.float32(self.raster[self.nodata_cell != True].min() - 2.0) if not one_hot else nan
        print(' ', self.mean, self.std, self.no_data)

        if np.sum(self.nodata_cell) > 0:
            self.raster[self.nodata_cell] = self.no_data

        if normalized and dtype != np.ubyte:
            self.raster = (self.raster - self.mean) / self.std
            if np.sum(self.nodata_cell) > 0:
                self.no_data = self.raster[self.nodata_cell][0] # update no_data
            '''
            # normalizing the whole raster given available data (therefore avoiding no_data)...
            self.raster[selected_cell] = (self.raster[selected_cell] - self.raster[selected_cell].mean()) \
                / self.raster[selected_cell].std()  # TODO all raster with nan or without nan
            '''
        if self.one_hot:
            if unique_values is not None: # user-defined values
                # NOTE: unique_values에는 nodata label(230)이 들어가지 않음. 즉, nodata는 입력 데이터로 구성되지 않음.
                self.unique_values = np.array(unique_values)
            else:
                # unique values for one-hot encoding
                self.unique_values = np.unique(self.raster[self.nodata_cell != True])

        # setting the shape of the raster
        self.shape = self.raster.shape

    def coords_to_index(self, x, y):
        #row_num = int(self.n_rows - (lat - self.y_min) / self.y_resolution)
        #col_num = int((long - self.x_min) / self.x_resolution)
        col_num, row_num = ~self.geospaital_transform * (x, y)

        return round(row_num), round(col_num)

    def index_to_coords(self, row_num, col_num):
        x, y = self.geospaital_transform * (col_num, row_num)
        return x, y

    def _get_patch(self, item, cancel_one_hot=False):
        """
        Avoid using this method directly

        :param item: the GPS position (latitude, longitude)
        :param cancel_one_hot: if True, one hot encoding will not be used
        :return: a patch
        """
        lat, long = item[0], item[1]
        row_num, col_num = self.coords_to_index(long, lat)
        H, W = self.shape[0], self.shape[1]

        # environmental vector
        if self.size == 1:
            patch = self.raster[row_num, col_num].astype(np.float32)
            if self.one_hot and not cancel_one_hot:
                patch = np.array([(patch == i).astype(np.float32) for i in self.unique_values])
            else:
                patch = patch[np.newaxis]
        # environmental tensor
        else:
            half_size = int(self.size/2)
            top, bottom = np.clip((row_num-half_size, row_num+half_size), a_min=0, a_max=H)
            left, right = np.clip((col_num-half_size, col_num+half_size), a_min=0, a_max=W)

            patch = self.raster[top:bottom, left:right].astype(np.float32)

            # padding
            pad_top, pad_bottom = (max(0, 0-(row_num-half_size)), max(0, (row_num+half_size-1)-(H-1)))
            pad_left, pad_right = (max(0, 0-(col_num-half_size)), max(0, (col_num+half_size-1)-(W-1)))

            patch = np.pad(patch, ((pad_top, pad_bottom), (pad_left, pad_right)), \
                            constant_values=self.no_data)

            if self.one_hot and not cancel_one_hot:
                patch = np.array([(patch == i).astype(np.float32) for i in self.unique_values])
            else:
                patch = patch[np.newaxis]

        return patch

    def _is_out_of_bounds(self, item):
        lat, long = item[0], item[1]
        row_num, col_num = self.coords_to_index(long, lat)
        H, W = self.shape[0], self.shape[1]

        if row_num < 0 or row_num > H-1 or col_num < 0 or col_num > W-1:
            return True
        else:
            return False

    def _is_nodata(self, item):
        lat, long = item[0], item[1]
        row_num, col_num = self.coords_to_index(long, lat)
        H, W = self.shape[0], self.shape[1]

        if self.size == 1:
            if self.nodata_cell[row_num, col_num] == True:
                return True
            else:
                return False
        else:
            half_size = int(self.size/2)
            top, bottom = np.clip((row_num-half_size, row_num+half_size), a_min=0, a_max=H)
            left, right = np.clip((col_num-half_size, col_num+half_size), a_min=0, a_max=W)

            # NOTE: tensor에서 no data를 padding하는 경우는 제외함. 나중에 다시 생각해보기.
            if np.sum(self.nodata_cell[top:bottom, left:right] == True) > 0:
                return True
            else:
                return False

    def __len__(self):
        """
        :return: the depth of the tensor/vector...
        """
        if self.one_hot:
            return int(self.unique_values.shape[0])
        else:
            return 1

    def __getitem__(self, item, cancel_one_hot=False):
        """
        The method to use to retrieve a patch.

        :param item: GPS position (latitude, longitude)
        :param cancel_one_hot: if true the one hot encoding representation will be disabled
        :return: the extracted patch with eventually some transformations
        """
        patch = self._get_patch(item, cancel_one_hot).copy()
        if self.transform:
            patch = self.transform(patch)

        return patch


class PatchExtractor(object):
    """
    PatchExtractor enables the extraction of an environmental tensor from multiple rasters given a GPS
    position.
    """
    def __init__(self, root_path, raster_metadata, size=64, verbose=False, resolution=1.):
        self.root_path = root_path
        self.raster_metadata = raster_metadata
        self.size = size

        self.verbose = verbose
        self.resolution = resolution

        self.rasters = []
        self.n_data_dims = 0

    def add_all(self, normalized=False, transform=None, ignore=[]):
        """
        Add all variables (rasters) available at root_path

        :param normalized: if True, each raster will be normalized
        :param transform: a function to apply on each patch
        :param ignore: a list to ignore specific rasters
        """
        for key in natsorted(self.raster_metadata.keys()):
            if (key in ignore) or ('ignore' in self.raster_metadata[key]):
                continue
            self.append(key, normalized=normalized, transform=transform)

        print(f'The number of environmental variables: {len(self.rasters)}')
        print(f'The size of a tensor/vector: {self.n_data_dims}\n')

    def append(self, raster_name, **kwargs):
        """
        This method append a new raster given its name

        :param raster_name:
        :param kwargs: nan, normalized, transform
        """
        # you may want to add rasters one by one if specific configuration are required on a per raster
        # basis
        print('Adding ratser: ' + raster_name, end='')
        params = {**self.raster_metadata[raster_name]}
        for k in kwargs.keys():
            if kwargs[k] != 'default':
                params[k] = kwargs[k]
        try:
            r = Raster(self.root_path + '/' + raster_name, size=self.size, **params)
            self.rasters.append(r)
            self.n_data_dims += len(r)
            print('')
        except rasterio.errors.RasterioIOError:
            print(' (not available...)')

    def remove_redundant_positions(self, raster_name, pos, drop_nodata=True):
        # NOTE: 반드시 raster 중 가장 해상도가 높은 레이어를 사용해야함.
        r = self.get_raster(raster_name)

        # remove redundancies
        rows_cols = [(r.coords_to_index(long, lat)) for lat, long in pos]
        rows_cols = np.unique(rows_cols, axis=0)
        n_red = len(pos) - len(rows_cols)

        n_oob = 0
        n_nodata = 0
        new_pos = []
        for row, col in rows_cols:
            long, lat = r.index_to_coords(row, col)
            if self._is_out_of_bounds((lat, long)):
                n_oob += 1
                continue
            if drop_nodata:
                if self._is_nodata((lat, long)):
                    n_nodata += 1
                    continue
            new_pos.append((lat, long))

        print(f'removed redundancies: {n_red}, out of bounds: {n_oob}, removed missing data: {n_nodata}\n')

        return np.array(new_pos)

    def get_valid_positions(self, raster_name, invalid_pos=None, buffer_pos=None, sample_size=0, drop_nodata=True, 
                                  exclusion_dist=0, raster_crs=4326, local_crs=5181):
        '''
        Return valid positions obtained by raster and invalid_pos

        :param invalid_pos: invalid positions
        :sample_size: the number of random sampling points
        '''
        # NOTE: globcover는 바다 지역이 no data가 아니라, water body라는 라벨로 정의되어 있음.
        # 따라서 샘플링 참조 지역으로 선택해선 안됨.
        assert raster_name != 'globcover', 'globcover is not supported.'
        
        # extract usable area
        r = self.get_raster(raster_name)
        valid_cell = (r.nodata_cell == False)
        if invalid_pos is not None:
            for lat, long in invalid_pos:
                row, col = r.coords_to_index(long, lat)
                valid_cell[row, col] = False

        # get valid rows and cols
        print(f'valid cell has {np.sum(valid_cell)} points.')
        rows, cols = np.nonzero(valid_cell)
        rows_cols_array = np.stack((rows, cols)).T

        # random sampling
        if sample_size > 0:
            assert sample_size < len(rows_cols_array), 'sample size is over the available raster points.'
            #len_p = int(len(invalid_pos)) if invalid_pos is not None else 0
            #sample_size = len_p if sample_size < len_p else sample_size
            rows_cols_array = rows_cols_array[np.random.choice(rows_cols_array.shape[0], sample_size, replace=False), :]
            print(f'random sampling size is {sample_size}.')

        sampled_pos = []
        for row, col in rows_cols_array:
            long, lat = r.index_to_coords(row, col) # slow calculation
            sampled_pos.append((lat, long))
        sampled_pos = np.array(sampled_pos)

        # filtering out sampled data
        # drop data within exclusion buffers
        n_eb = 0
        if exclusion_dist > 0:
            assert buffer_pos is not None, "exclusion buffer should use 'buffer_pos'."
            crs_from = raster_crs # epsg:4326 (WGS84, GPS 좌표계)
            crs_to = local_crs # epsg:5181 (카카오맵 좌표계) 왜곡이 적은 로컬 좌표계를 사용해야함.
            print(f'exclusion distance sets to {exclusion_dist}m.')
            print(f'epsg:{crs_to} CRS is used for reprojection.')

            # 점 간의 거리를 미터 단위로 정확히 계산할 수 있는 좌표계로 reprojection
            sampled_xy_pos = [Point(lon, lat) for lat, lon in sampled_pos]
            reprojected_points = gpd.GeoDataFrame(geometry=sampled_xy_pos, crs=crs_from).to_crs(crs_to)

            # circle buffer 생성
            buffer_xy_pos = [Point(lon, lat) for lat, lon in buffer_pos]
            buffer_geometries = gpd.GeoSeries(buffer_xy_pos, crs=crs_from).to_crs(crs_to).buffer(exclusion_dist)
            circle_buffers = gpd.GeoDataFrame(geometry=buffer_geometries)

            # circle buffer에 포함된 점 제거
            # NOTE 'pygeos' library is used.
            within_circles = gpd.sjoin(reprojected_points, circle_buffers, predicate='within', how='left')
            out_of_circles = within_circles[within_circles['index_right'].isna()]
            n_eb = len(sampled_pos) - len(out_of_circles)
            sampled_pos = sampled_pos[out_of_circles.index.to_numpy()]

        n_oob = 0
        n_nodata = 0
        valid_pos = []
        for lat, long in sampled_pos:
            # drop out of bounds data by checking other rasters
            if self._is_out_of_bounds((lat, long)):
                n_oob += 1
                continue
            # drop no data by checking other rasters
            if drop_nodata:
                if self._is_nodata((lat, long)):
                    n_nodata += 1
                    continue

            valid_pos.append((lat, long))

        print(f'selected samples: {len(valid_pos)}, out of bounds: {n_oob}, '
              f'removed missing data: {n_nodata}, excluded data by exclusion distance: {n_eb}\n')

        return np.array(valid_pos)

    def _is_out_of_bounds(self, item):
        for r in self.rasters:
            if r._is_out_of_bounds(item):
                return True

        return False

    def _is_nodata(self, item):
        for r in self.rasters:
            if r._is_nodata(item):
                return True

        return False

    def get_raster(self, raster_name):
        for r in self.rasters:
            if r.name == raster_name:
                return r

        assert False, 'no valid raster found.'

    def get_raster_names(self):
        return [r.name for r in self.rasters]

    def clean(self):
        """
        Remove all rasters from the extractor.
        """

        print('Removing all rasters...')
        self.rasters = []
        self.n_data_dims = 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        result = ''
        for r in self.rasters:
            result += '-' * 50 + '\n'
            result += 'title: ' + r.name + '\n'
            #result += '\t x_min: ' + str(r.x_min) + '\n'
            #result += '\t y_min: ' + str(r.y_min) + '\n'
            #result += '\t x_resolution: ' + str(r.x_resolution) + '\n'
            #result += '\t y_resolution: ' + str(r.y_resolution) + '\n'
            #result += '\t n_rows: ' + str(r.n_rows) + '\n'
            #result += '\t n_cols: ' + str(r.n_cols) + '\n'

        return result

    def __getitem__(self, item, cancel_one_hot=False):
        """
        :param item: the GPS location (latitude, longitude)
        :return: return the environmental tensor or vector (size>1 or size=1)
        """
        return np.concatenate([r.__getitem__(item, cancel_one_hot) for r in self.rasters])

    def __len__(self):
        """
        :return: the number of variables (not the size of the tensor when some variables have a one hot encoding
                 representation)
        """
        return len(self.rasters)