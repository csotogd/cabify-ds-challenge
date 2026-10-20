import unittest
import numpy as np
from scipy.sparse import csr_matrix
from rutas.graph_extension import weighted_overlap,to_target_distances,distance_summary,sample_observed,summarize_match


class GraphMeaningTests(unittest.TestCase):
    def test_one_way_distance_is_source_to_target(self):
        graph=csr_matrix(([100.,200.],([0,1],[1,2])),shape=(4,4))
        d=to_target_distances(graph.T.tocsr(),[2])
        np.testing.assert_allclose(d[:3],[300,200,0])
        self.assertTrue(np.isinf(d[3]))
        back=to_target_distances(graph.T.tocsr(),[0])
        self.assertTrue(np.isinf(back[2]))

    def test_directed_overlap_does_not_conflate_reverse_travel(self):
        a={(1,2):100.,(2,3):200.}
        self.assertEqual(weighted_overlap(a,a),1.)
        self.assertEqual(weighted_overlap(a,{(2,1):100.,(3,2):200.}),0.)
        self.assertAlmostEqual(weighted_overlap(a,{(1,2):100.}),1/3)

    def test_censoring_is_explicit_and_lengths_are_weights(self):
        s=distance_summary([0.,np.inf],[9.,1.],1000.)
        self.assertAlmostEqual(s['mean_capped_m'],1000.)
        self.assertAlmostEqual(s['fraction_unreachable_or_over_10km'],.1)
        self.assertAlmostEqual(s['fraction_over_50m'],.1)

    def test_sampling_uses_observations_and_preserves_ends(self):
        p=np.column_stack([np.full(1000,40.),np.linspace(-3.7,-3.6,1000)])
        sampled=sample_observed(p,40)
        self.assertLessEqual(len(sampled),40)
        np.testing.assert_equal(sampled[[0,-1]],p[[0,-1]])
        self.assertTrue(set(map(tuple,sampled)).issubset(set(map(tuple,p))))

    def test_partial_edges_are_not_counted_as_full_edges_per_gps_fix(self):
        raw=dict(input_points=[[0,0]]*3,tracepoints=[{}]*3,matchings=[dict(distance=10,duration=2,confidence=1,
            legs=[dict(annotation=dict(nodes=[1,2],distance=[3.])),dict(annotation=dict(nodes=[1,2],distance=[7.]))])])
        f,edges,weights=summarize_match(raw,{1:0,2:1},np.array([[0.,0.],[.001,0.]]))
        self.assertAlmostEqual(sum(weights.values()),10.)
        self.assertEqual(len(edges),1)


if __name__=='__main__':unittest.main()
