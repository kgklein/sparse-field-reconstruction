import numpy as np
import pytest
from sparse_recon.sampling.geometries import (
    generate_sampling_points,
    random_points_in_box,
    clustered_points,
    tetrahedron_like,
    clustered_points_in_box,
    multi_probe_like_points_2d,
    generate_flyby_points,
)

def test_random_points_in_box():
    points = random_points_in_box(n_points=10, dim=2, seed=0)
    assert points.shape == (10, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)

def test_clustered_points():
    center = np.array([0.5, 0.5])
    offsets = np.array([[0.1, 0.1], [-0.1, -0.1]])
    points = clustered_points(center, offsets)
    assert points.shape == (2, 2)
    assert np.allclose(points[0], [0.6, 0.6])
    assert np.allclose(points[1], [0.4, 0.4])

def test_tetrahedron_like():
    points = tetrahedron_like()
    assert points.shape == (4, 3)
    expected_points = np.array([
        [0.6, 0.6, 0.6],
        [0.6, 0.4, 0.4],
        [0.4, 0.6, 0.4],
        [0.4, 0.4, 0.6],
    ])
    assert np.allclose(points, expected_points)

def test_clustered_points_in_box():
    points = clustered_points_in_box(n_points=20, dim=2, seed=0)
    assert points.shape == (20, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)

def test_multi_probe_like_points_2d():
    points = multi_probe_like_points_2d(n_points=10, seed=0)
    assert points.shape == (10, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)

def test_generate_flyby_points_2d():
    points = generate_flyby_points(n_points=10, dim=2, seed=0)
    assert points.shape == (10, 2)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)
    # Check if points are collinear
    n_points_2d = 10 # Define n_points locally
    if n_points_2d > 1:
        diffs = np.diff(points, axis=0)
        # All diffs should be parallel, i.e., cross product should be zero (or very small)
        # For 2D, this means the determinant of any two diff vectors should be zero
        if len(diffs) > 1:
            # For 2D, if points are collinear, the y/x ratio between consecutive points should be constant.
            # Or, more robustly, the area of the triangle formed by three consecutive points should be zero.
            # This can be checked using the determinant of vectors (p2-p1) and (p3-p1).
            # For a line, if p1, p2, p3 are points, then (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1) should be 0.
            for i in range(len(points) - 2):
                p1 = points[i]
                p2 = points[i+1]
                p3 = points[i+2]
                # Check if (p2-p1) and (p3-p1) are parallel by checking if the determinant is close to zero
                det = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
                assert np.isclose(det, 0.0), f"Points are not collinear in 2D: {points[i:i+3]}"

def test_generate_flyby_points_3d():
    n_points_3d = 10 # Define n_points locally
    points = generate_flyby_points(n_points=n_points_3d, dim=3, seed=0)
    assert points.shape == (n_points_3d, 3)
    assert np.all(points >= 0.0)
    assert np.all(points <= 1.0)
    # Check if points are collinear
    if n_points_3d > 1:
        # For 3D, cross product of (p2-p1) and (p3-p1) should be a zero vector
        for i in range(len(points) - 2):
            p1 = points[i]
            p2 = points[i+1]
            p3 = points[i+2]
            vec1 = p2 - p1
            vec2 = p3 - p1
            cross_prod = np.cross(vec1, vec2)
            assert np.allclose(cross_prod, 0.0), f"Points are not collinear in 3D: {points[i:i+3]}"

def test_generate_sampling_points_random():
    points = generate_sampling_points("random", 10, 2, seed=0)
    assert points.shape == (10, 2)

def test_generate_sampling_points_clustered():
    points = generate_sampling_points("clustered", 10, 2, seed=0)
    assert points.shape == (10, 2)

def test_generate_sampling_points_multi_probe_like():
    points = generate_sampling_points("multi_probe_like", 10, 2, seed=0)
    assert points.shape == (10, 2)

def test_generate_sampling_points_tetrahedron_like():
    points = generate_sampling_points("tetrahedron_like", 4, 3, seed=0)
    assert points.shape == (4, 3)

def test_generate_sampling_points_flyby_2d():
    points = generate_sampling_points("flyby", 10, 2, seed=0)
    assert points.shape == (10, 2)

def test_generate_sampling_points_flyby_3d():
    points = generate_sampling_points("flyby", 10, 3, seed=0)
    assert points.shape == (10, 3)

def test_generate_sampling_points_unknown_geometry():
    with pytest.raises(ValueError, match="Unknown geometry"):
        generate_sampling_points("unknown", 10, 2)

def test_generate_sampling_points_tetrahedron_like_wrong_dim():
    with pytest.raises(ValueError, match="tetrahedron_like geometry currently supports dim=3 only"):
        generate_sampling_points("tetrahedron_like", 4, 2)

def test_generate_sampling_points_flyby_wrong_dim():
    with pytest.raises(ValueError, match="Flyby geometry not supported for dimension"): # Adjust regex if needed
        generate_sampling_points("flyby", 10, 4)
