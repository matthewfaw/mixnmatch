from sklearn.decomposition import PCA
import scipy as sp
import scipy.spatial
import numpy as np
from matplotlib import pyplot as plt


class PartitionStrategy:
    def __init__(self, dim):
        self.dim = dim


class CoordinateHalvingPartitioningStrategy(PartitionStrategy):

    def get_centroid(self, points):
        return points.mean(axis=0)

    def _partition_2d(self, points):
        centroid = self.get_centroid(points)
        children = [
            np.vstack((points[0,:], centroid)),
            np.vstack((points[1,:], centroid)),
        ]
        return children

    def _partition_3dplus(self, points):
        # Project to lower dim
        pca = PCA(n_components=self.dim - 1)
        face = pca.fit_transform(points)

        # Split in half along random axis
        centroid = self.get_centroid(face)
        ## Get convex hull of points, which will give the equations defining the polytope
        hull = scipy.spatial.ConvexHull(face)
        dim_to_split = np.random.randint(face.shape[1])
        dir = np.zeros(face.shape[1])
        small_mult = 0.001
        dir[dim_to_split] = 1
        ## Define the new inequality of form [a, b] which is used as an inequality a^T x + b <= 0
        new_eqn = np.zeros(face.shape[1] + 1)
        new_eqn[dim_to_split] = 1
        new_eqn[-1] = -centroid[dim_to_split]
        ## Find one of the new polytope after adding the new partitioning inequality
        halfspaces_upper = np.vstack((hull.equations, -new_eqn))
        mult_upper = np.max(face[:,dim_to_split]) - centroid[dim_to_split]
        interior_pt_upper = centroid + small_mult * mult_upper * dir
        hi_upper = scipy.spatial.HalfspaceIntersection(halfspaces_upper, interior_pt_upper)
        ## Find the other new polytope after adding the new partitioning inequality
        halfspaces_lower = np.vstack((hull.equations, new_eqn))
        mult_lower = centroid[dim_to_split] - np.min(face[:,dim_to_split])
        interior_pt_lower = centroid - small_mult * mult_lower * dir
        hi_lower = scipy.spatial.HalfspaceIntersection(halfspaces_lower, interior_pt_lower)

        # Project results back to simplex
        return [
            pca.inverse_transform(hi_upper.intersections),
            pca.inverse_transform(hi_lower.intersections)
        ]

    def partition(self, points):
        if self.dim >= 3:
            return self._partition_3dplus(points)
        elif self.dim == 2:
            return self._partition_2d(points)
        else:
            assert False


class DelaunayPartitioningStrategy(PartitionStrategy):

    def get_centroid(self, points):
        return points.mean(axis=0)

    def partition_3dplus_simplified(self, points, plot_pts):
        # Project face to n-1 space
        pca = PCA(n_components=self.dim - 1)
        face = pca.fit_transform(points)
        centroid = self.get_centroid(face)
        partitioned_face = np.vstack((face, centroid))
        # Triangulate the face + centroid to get partitioned face
        tri_child = sp.spatial.Delaunay(partitioned_face)
        # Project the points back to original space
        new_points = pca.inverse_transform(partitioned_face)

        if plot_pts:
            self.plot_projected_simplex(partitioned_face)
            self.plot_simplex(new_points)

        return tri_child.simplices, new_points

    def partition_2d(self, points, plot_pts):
        centroid = self.get_centroid(points)
        new_points = np.vstack((points, centroid))
        simplices = [[0, 2], [1, 2]]
        return simplices, new_points

    def partition(self, points, plot_pts=False):
        if self.dim >= 3:
            simplices, new_points = self.partition_3dplus_simplified(points, plot_pts)
        elif self.dim == 2:
            simplices, new_points = self.partition_2d(points, plot_pts)
        else:
            assert False
        children = [new_points[s] for s in simplices]
        return children

    def plot_projected_simplex(self, points):
        tri = sp.spatial.Delaunay(points)
        plt.triplot(points[:, 0], points[:, 1], tri.simplices)
        for j, s in enumerate(tri.simplices):
            p = self.get_centroid(s, points)
            plt.text(p[0], p[1], '#%d' % (j), ha='center')
        plt.show()

    def plot_simplex(self, points):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='s')
        ax.scatter([1, 0, 0], [0, 1, 0], [0, 0, 1], c='b', marker='o')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.view_init(45, 45)
        plt.show()


class ConstantPartitioningStrategy(PartitionStrategy):
    def __init__(self, dim, simplex_point=None):
        super().__init__(dim)
        if simplex_point is None:
            simplex_point = np.ones(self.dim) / self.dim
        assert all(entry >= 0 for entry in simplex_point)
        # assert np.sum(simplex_point) == 1
        self.simplex_point = simplex_point

    def get_centroid(self, points):
        return self.simplex_point
