"""Known local deformations must recover dimensions a whole-body scale cannot."""
import unittest
import numpy as np
from scripts.known_axial_profile_rectify import warp_points, warp_with_jacobian, fit_profile, radius, center


class LocalProfileCorrection(unittest.TestCase):
    def test_jacobian_and_order(self):
        g = dict(knots=[-1., -.7, .7, 1.], radii=[.27, .20, .26],
                 widths=[.03, .03], center=[.02, -.03], slope=[.01, -.005])
        a = dict(knots=[-.92, -.84, .84, 1.], radii=[.225, .19, .35],
                 widths=[.01, .01], center=[0., 0.], slope=[0., 0.])
        points = np.array([[-.71, .15, -.17], [0., .22, .04], [.69, -.15, .17]])
        out, jac = warp_with_jacobian(points, g, a)
        for k in range(3):
            delta = np.zeros_like(points); delta[:, k] = 1e-6
            numerical = (warp_points(points+delta, g, a)-warp_points(points-delta, g, a))/2e-6
            np.testing.assert_allclose(jac[:, :, k], numerical, atol=1e-7)
        self.assertTrue(np.all(np.linalg.det(jac) > 0))
        bad = {**g, 'knots': [0., 1., .5, 2.]}
        with self.assertRaisesRegex(ValueError, 'Ordered'):
            warp_points(points, bad, a)

    def test_recover_generated_radii_and_seams_from_surface(self):
        import trimesh
        generated = dict(knots=[-1., -.72, .66, 1.1], radii=[.27, .20, .26],
                         widths=[.025, .025], center=[.02, -.03], slope=[.01, -.005])
        authored = dict(knots=[-.92, -.84, .84, 1.], radii=[.225, .19, .35],
                        widths=[.015, .015], center=[0., 0.], slope=[0., 0.])
        xs = np.linspace(-1., 1.1, 128); angles = np.arange(48)*2*np.pi/48
        yz = center(generated, xs)[:, None, :] + radius(generated, xs)[:, None, None]*np.array([np.cos(angles), np.sin(angles)]).T
        vertices = np.concatenate([np.broadcast_to(xs[:, None, None], (128, 48, 1)), yz], axis=2).reshape(-1, 3)
        faces = []
        for i in range(127):
            for j in range(48):
                k, n = i*48+j, i*48+(j+1)%48
                faces += [[k, n, n+48], [k, n+48, k+48]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        fitted, report = fit_profile(mesh, authored)
        np.testing.assert_allclose(fitted['radii'], generated['radii'], atol=.001)
        np.testing.assert_allclose(fitted['knots'][1:3], generated['knots'][1:3], atol=.005)
        expected = warp_points(vertices, generated, authored)
        actual = warp_points(vertices, fitted, authored)
        self.assertLess(np.max(np.linalg.norm(actual-expected, axis=1)), .005)
        self.assertEqual(report['triangleCount'], len(faces))

    def test_independent_head_shaft_and_axial_correspondence(self):
        generated = dict(knots=[-1., -.78, .72, 1.1], radii=[.27, .20, .26],
                         widths=[.004, .004], center=[.02, -.03], slope=[.01, -.005])
        authored = dict(knots=[-.92, -.84, .84, 1.], radii=[.225, .19, .35],
                        widths=[.004, .004], center=[0., 0.], slope=[0., 0.])
        x = np.array([-.95, 0., .98])
        r = np.array(generated['radii'])
        centers = np.array(generated['center']) + x[:, None] * generated['slope']
        points = np.column_stack([x, centers + np.column_stack([r + .002, np.zeros(3)])])
        out = warp_points(points, generated, authored)
        self.assertAlmostEqual(out[1, 1], .192, delta=.0002,
                               msg='The shaft must retain its small surface relief at its authored radius')
        self.assertAlmostEqual(out[2, 1], .352692307692, delta=.0002,
                               msg='The undersized head needs its own correction, independent of the shaft')
        self.assertTrue(np.all(np.diff(out[:, 0]) > 0), 'Axial correspondence must remain monotone')
        self.assertLess(abs(out[:, 2]).max(), 1e-9, 'Recovered centerline must reach the authored axis')


if __name__ == '__main__':
    unittest.main()
