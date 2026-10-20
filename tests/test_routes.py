import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from rutas.features import features_xy
from rutas.data import load_data

class GeometryTests(unittest.TestCase):
    def test_identical_and_sampling_invariance(self):
        a=np.array([[0.,0.],[1000.,0.]])
        b=np.column_stack([np.linspace(0,1000,101),np.zeros(101)])
        f=features_xy(a,b)
        self.assertAlmostEqual(f['mean_est_to_real_m'],0)
        self.assertAlmostEqual(f['relative_length_delta'],0)
        self.assertAlmostEqual(f['buffer_iou_50m'],1)
    def test_parallel_offset_and_thresholds(self):
        a=np.array([[0.,0.],[1000.,0.]])
        f=features_xy(a,a+[0,75])
        self.assertAlmostEqual(f['mean_est_to_real_m'],75)
        self.assertEqual(f['fraction_est_to_real_over_50m'],1)
        self.assertEqual(f['fraction_est_to_real_over_100m'],0)
        self.assertAlmostEqual(f['frechet_64_m'],75)
    def test_reversed_order_not_hidden_by_overlap(self):
        a=np.array([[0.,0.],[1000.,0.]])
        f=features_xy(a,a[::-1])
        self.assertEqual(f['hausdorff_discrete_m'],0)
        self.assertAlmostEqual(f['frechet_64_m'],1000)
    def test_split_and_no_duplicate_routes(self):
        df,_=load_data()
        self.assertEqual(df.split.value_counts().to_dict(),{'train':1432,'test':477,'val':477})
        self.assertTrue(df.route_hash.is_unique)
        self.assertTrue(df.index.is_unique)
        self.assertTrue((df.n_votes<=df.n_unique_annotators).all())
        self.assertEqual(int(df.n_conflicted_annotators.sum()),7)

    def test_encoder_training_excludes_validation_and_test(self):
        import json
        from rutas.data import ROOT
        path=ROOT/'resultados/rutas/t2vec_manifest.json'
        if not path.exists():self.skipTest('Preentrenamiento pendiente')
        manifest=json.loads(path.read_text());df,_=load_data()
        fit=set(manifest['pretrained_on_journeys']);inner=set(manifest['inner_validation_journeys'])
        self.assertFalse(fit&inner)
        self.assertEqual(fit|inner,set(df.index[df.split=='train']))

    def test_prediction_probabilities_are_valid(self):
        import json
        from rutas.data import ROOT
        path=ROOT/'resultados/rutas/predictions.json'
        if not path.exists():self.skipTest('Entrenamiento pendiente')
        rows=json.loads(path.read_text());self.assertEqual(len(rows),2386)
        for row in rows:
            self.assertTrue(all(np.isfinite(p) and 0<=p<=1 for p in row['probabilities'].values()))

if __name__=='__main__': unittest.main()
