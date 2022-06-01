import numpy as np
from tqdm import tqdm
import math
import copy, cv2

def uniform_blend(img1, img2):
    # grayscale
    gray1 = np.mean(img1, axis=-1)
    gray2 = np.mean(img2, axis=-1)
    result = (img1.astype(np.float64) + img2.astype(np.float64))

    g1, g2 = gray1 > 0, gray2 > 0
    g = g1 & g2
    mask = np.expand_dims(g * 0.5, axis=-1)
    mask = np.tile(mask, [1, 1, 3])
    mask[mask == 0] = 1
    result *= mask
    result = result.astype(np.uint8)

    return result

def final_size(src_img, dst_img, project_H):
    """
    get the size of stretched (stitched) image
    :param src_img: source image
    :param dst_img: destination image
    :param project_H: global homography
    :return:
    """
    h, w, c = src_img.shape

    corners = []
    pt_list = [np.array([0, 0, 1], dtype=np.float64), np.array([0, h, 1], dtype=np.float64),
               np.array([w, 0, 1], dtype=np.float64), np.array([w, h, 1], dtype=np.float64)]

    for pt in pt_list:
        vec = np.matmul(project_H, pt)
        x, y = vec[0] / vec[2], vec[1] / vec[2]
        corners.append([x, y])

    corners = np.array(corners).astype(np.int)

    h, w, c = dst_img.shape

    max_x = max(np.max(corners[:, 0]), w)
    max_y = max(np.max(corners[:, 1]), h)
    min_x = min(np.min(corners[:, 0]), 0)
    min_y = min(np.min(corners[:, 1]), 0)

    width = max_x - min_x
    height = max_y - min_y
    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    return width, height, offset_x, offset_y

def get_mesh(size, mesh_size, start=0):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param start: default 0
    :return:
    """
    w, h = size
    x = np.linspace(start, w, mesh_size)
    y = np.linspace(start, h, mesh_size)

    return np.stack([x, y], axis=0)


def get_vertice(size, mesh_size, offsets):
    """
    :param size: final size [width, height]
    :param mesh_size: # of mesh
    :param offsets: [offset_x, offset_y]
    :return:
    """
    w, h = size
    x = np.linspace(0, w, mesh_size)
    y = np.linspace(0, h, mesh_size)
    next_x = x + w / (mesh_size * 2)
    next_y = y + h / (mesh_size * 2)
    next_x, next_y = np.meshgrid(next_x, next_y)
    vertices = np.stack([next_x, next_y], axis=-1)
    vertices -= np.array(offsets)

    return vertices

class Apap(object):
    def __init__(self, final_size, offset):
        """
        :param opt: Engine running Options
        :param final_size: final result (Stitched) image size
        :param offset: The extent to which the image is stretched
        """
        super().__init__()
        self.gamma = 0.0001
        self.sigma = 8.5
        self.final_width, self.final_height = final_size
        self.offset_x, self.offset_y = offset

    def global_homography(self, src_point, dst_point):
        raise NotImplementedError

    def local_homography(self, src_point, dst_point, vertices):
        """
        local homography estimation
        :param src_point: shape [sample_n, 2]
        :param dst_point:
        :param vertices: shape [mesh_size, mesh_size, 2]
        :return: np.ndarray [meshsize, meshsize, 3, 3]
        """
        sample_n, _ = src_point.shape
        mesh_n, pt_size, _ = vertices.shape

        N1, nf1 = self.getNormalize2DPts(src_point)
        N2, nf2 = self.getNormalize2DPts(dst_point)

        C1 = self.getConditionerFromPts(nf1)
        C2 = self.getConditionerFromPts(nf2)

        cf1 = self.point_normalize(nf1, C1)
        cf2 = self.point_normalize(nf2, C2)

        inverse_sigma = 1. / (self.sigma ** 2)
        local_homography_ = np.zeros([mesh_n, pt_size, 3, 3], dtype=np.float)
        local_weight = np.zeros([mesh_n, pt_size, sample_n])
        aa = self.matrix_generate(sample_n, cf1, cf2)  # initiate A

        for i in range(mesh_n):
            for j in range(pt_size):
                dist = np.tile(vertices[i, j], (sample_n, 1)) - src_point
                weight = np.exp(-(np.sqrt(dist[:, 0] ** 2 + dist[:, 1] ** 2) * inverse_sigma))
                weight[weight < self.gamma] = self.gamma
                local_weight[i, j, :] = weight
                A = np.expand_dims(np.repeat(weight, 2), -1) * aa
                W, U, V = cv2.SVDecomp(A)
                h = V[-1, :]
                h = h.reshape((3, 3))
                h = np.linalg.inv(C2).dot(h).dot(C1)
                h = np.linalg.inv(N2).dot(h).dot(N1)
                h = h / h[2, 2]
                local_homography_[i, j] = h
        return local_homography_, local_weight

    @staticmethod
    def warp_coordinate_estimate(pt, homography):
        """
        source points -> target points matrix multiplication with homography
        [ h11 h12 h13 ] [x]   [x']
        [ h21 h22 h23 ] [y] = [y']
        [ h31 h32 h33 ] [1]   [s']
        :param pt: source point
        :param homography: transfer relationship
        :return: target point
        """
        target = np.matmul(homography, pt)
        target /= target[2]
        return target

    def local_warp(self, ori_img: np.ndarray, local_homography: np.ndarray, mesh: np.ndarray,
                   progress=False) -> np.ndarray:
        """
        this method requires improvement with the numpy algorithm (because of speed)

        :param ori_img: original input image
        :param local_homography: [mesh_n, pt_size, 3, 3] local homographies np.ndarray
        :param mesh: [2, mesh_n+1]
        :param progress: print warping progress or not.
        :return: result(warped) image
        """
        mesh_w, mesh_h = mesh
        ori_h, ori_w, _ = ori_img.shape
        warped_img = np.zeros([self.final_height, self.final_width, 3], dtype=np.uint8)

        for i in tqdm(range(self.final_height)) if progress else range(self.final_height):
            m = np.where(i < mesh_h)[0][0]
            for j in range(self.final_width):
                n = np.where(j < mesh_w)[0][0]
                homography = np.linalg.inv(local_homography[m-1, n-1, :])
                x, y = j - self.offset_x, i - self.offset_y
                source_pts = np.array([x, y, 1])
                target_pts = self.warp_coordinate_estimate(source_pts, homography)
                if 0 < target_pts[0] < ori_w and 0 < target_pts[1] < ori_h:
                    warped_img[i, j, :] = ori_img[int(target_pts[1]), int(target_pts[0]), :]

        return warped_img

    @staticmethod
    def getNormalize2DPts(point):
        """
        :param point: [sample_num, 2]
        :return:
        """
        sample_n, _ = point.shape
        origin_point = copy.deepcopy(point)
        # np.ones(6) [1, 1, 1, 1, 1, 1]
        padding = np.ones(sample_n, dtype=np.float)
        c = np.mean(point, axis=0)
        pt = point - c
        pt_square = np.square(pt)
        pt_sum = np.sum(pt_square, axis=1)
        pt_mean = np.mean(np.sqrt(pt_sum))
        scale = math.sqrt(2) / (pt_mean + 1e-8)
        # https://www.programmersought.com/article/9488862074/
        t = np.array([[scale, 0, -scale * c[0]],
                      [0, scale, -scale * c[1]],
                      [0, 0, 1]], dtype=np.float)
        origin_point = np.column_stack((origin_point, padding))
        new_point = t.dot(origin_point.T)
        new_point = new_point.T[:, :2]
        return t, new_point

    @staticmethod
    def getConditionerFromPts(point):
        sample_n, _ = point.shape
        calculate = np.expand_dims(point, 0)
        mean_pts, std_pts = cv2.meanStdDev(calculate)
        mean_x, mean_y = np.squeeze(mean_pts)
        std_pts = np.squeeze(std_pts)
        std_pts = std_pts * std_pts * sample_n / (sample_n - 1)
        std_pts = np.sqrt(std_pts)
        std_x, std_y = std_pts
        std_x = std_x + (std_x == 0)
        std_y = std_y + (std_y == 0)
        norm_x = math.sqrt(2) / std_x
        norm_y = math.sqrt(2) / std_y
        T = np.array([[norm_x, 0, (-norm_x * mean_x)],
                      [0, norm_y, (-norm_y * mean_y)],
                      [0, 0, 1]], dtype=np.float)

        return T

    @staticmethod
    def point_normalize(nf, c):
        sample_n, _ = nf.shape
        cf = np.zeros_like(nf)

        for i in range(sample_n):
            cf[i, 0] = nf[i, 0] * c[0, 0] + c[0, 2]
            cf[i, 1] = nf[i, 1] * c[1, 1] + c[1, 2]

        return cf

    @staticmethod
    def matrix_generate(sample_n, cf1, cf2):
        A = np.zeros([sample_n * 2, 9], dtype=np.float)
        for k in range(sample_n):
            A[2 * k, 0] = cf1[k, 0]
            A[2 * k, 1] = cf1[k, 1]
            A[2 * k, 2] = 1
            A[2 * k, 6] = (-cf2[k, 0]) * cf1[k, 0]
            A[2 * k, 7] = (-cf2[k, 0]) * cf1[k, 1]
            A[2 * k, 8] = (-cf2[k, 0])

            A[2 * k + 1, 3] = cf1[k, 0]
            A[2 * k + 1, 4] = cf1[k, 1]
            A[2 * k + 1, 5] = 1
            A[2 * k + 1, 6] = (-cf2[k, 1]) * cf1[k, 0]
            A[2 * k + 1, 7] = (-cf2[k, 1]) * cf1[k, 1]
            A[2 * k + 1, 8] = (-cf2[k, 1])
        return A