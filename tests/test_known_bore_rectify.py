import unittest
import numpy as np
from scripts.known_bore_rectify import warp_with_jacobian, fit_bore_ellipse


class KnownBoreCorrectionTests(unittest.TestCase):
    def test_bore_center_radius_and_distant_detail(self):
        theta = np.linspace(0, 2*np.pi, 128, endpoint=False)
        p = np.column_stack([np.zeros(len(theta)), 1.27+.24*np.cos(theta), -.03+.24*np.sin(theta)])
        anchors = [dict(x=0., center=[1.27, -.03], radius=.24,
                        targetCenter=[1.25, 0.], targetRadius=.21)]
        q, jac = warp_with_jacobian(p, anchors, .44)
        np.testing.assert_allclose(np.linalg.norm(q[:, 1:]-[1.25, 0], axis=1), .21, atol=1e-12)
        far = np.array([[0., 0., 0.], [0., 1.25, -1.]])
        np.testing.assert_array_equal(warp_with_jacobian(far, anchors, .44)[0], far)
        self.assertTrue(np.all(np.linalg.det(jac) > 0))

    def test_jacobian_and_two_lug_centerline(self):
        anchors = [dict(x=x, center=c, radius=r, targetCenter=[1.25, 0.], targetRadius=.205)
                   for x,c,r in [(-.6,[1.22,-.01],.22),(.6,[1.26,.02],.24)]]
        p = np.array([[-.6,1.22+.22,-.01],[.6,1.26+.24,.02],[.1,1.57,.04]])
        q, jac = warp_with_jacobian(p, anchors, .48)
        np.testing.assert_allclose(q[:2, 1:], [[1.455,0],[1.455,0]], atol=1e-12)
        # Compare away from interpolation knots; derivative is piecewise.
        p = p[2:]; _, jac = warp_with_jacobian(p, anchors, .48)
        numeric = np.empty_like(jac)
        for i in range(3):
            h = np.zeros_like(p); h[:,i]=1e-6
            numeric[:,:,i]=(warp_with_jacobian(p+h, anchors, .48)[0]-warp_with_jacobian(p-h, anchors, .48)[0])/2e-6
        np.testing.assert_allclose(jac,numeric,atol=1e-7)

    def test_invalid_correspondence_rejected(self):
        p = np.array([[0.,1.5,0.]])
        for a in [[], [dict(x=0.,center=[1.25,0],radius=0.,targetCenter=[1.25,0],targetRadius=.2)],
                  [dict(x=0.,center=[1.25,0],radius=.24,targetCenter=[1.25,0],targetRadius=.2)]*2]:
            with self.assertRaises(ValueError):
                warp_with_jacobian(p,a,.4)

    def test_oval_bore_corrects_both_axes(self):
        theta=np.linspace(0,2*np.pi,256,endpoint=False)
        p=np.column_stack([.24*np.cos(theta)+1.27,.18*np.sin(theta)-.03])
        lines=np.stack([p,np.roll(p,-1,axis=0)],axis=1)
        observed=fit_bore_ellipse(lines,.21)
        anchor=dict(x=0.,**observed,targetCenter=[1.25,0.],targetRadius=.21)
        points=np.column_stack([np.zeros(len(p)),p])
        result,jac=warp_with_jacobian(points,[anchor],.44)
        np.testing.assert_allclose(np.linalg.norm(result[:,1:]-[1.25,0],axis=1),.21,atol=1e-10)
        self.assertTrue(np.all(np.linalg.det(jac)>0))


if __name__ == '__main__':
    unittest.main()
