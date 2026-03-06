__author__ = "Jianfeng Sun"
__version__ = "v1.0"
__copyright__ = "Copyright 2025"
__license__ = "MIT"
__email__ = "jianfeng.sunmt@gmail.com"
__maintainer__ = "Jianfeng Sun"

import numpy as np
import pandas as pd
import onnxruntime as ort
from functools import lru_cache
from drutai.util.Console import Console


def _session_options():
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 4
    return opts


@lru_cache(maxsize=None)
def _get_session(onnx_path):
    return ort.InferenceSession(onnx_path, sess_options=_session_options(), providers=['CPUExecutionProvider'])


def drestruct(data, met):
    if met == 'ConvMixer64':
        return np.reshape(data, [-1, 108, 108, 1])
    else:
        return data


class Model:

    def __init__(
            self,
            method,
            model_fp,
            mat_np,
            sv_fpn,
            batch_size=100,
            thres=0.5,
            verbose=True,
    ):
        self.method = method
        self.model_fp = model_fp
        self.mat_np = mat_np
        self.sv_fpn = sv_fpn
        self.batch_size = batch_size
        self.thres = thres

        self.console = Console()
        self.console.verbose = verbose

    def predict(self):
        session = _get_session(self.model_fp + '.onnx')
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        data = drestruct(self.mat_np, self.method).astype(np.float32)
        pred = session.run([output_name], {input_name: data})[0]
        df = pd.DataFrame(pred[:, [1]], columns=['prob_inter'])
        df['pred_type'] = df['prob_inter'].apply(lambda x: 'Interaction' if x > self.thres else 'Non-interaction')
        self.console.print("Predictions:\n{}".format(df))
        if self.sv_fpn:
            df.to_csv(
                self.sv_fpn,
                sep='\t',
                header=True,
                index=False,
            )
        return df

    # Aliases for backward compatibility
    m1 = predict
    m2 = predict


if __name__ == "__main__":
    p = Model(
        method='Seq2Seq',
        model_fp='model/seq2seq/seq2seq',
        mat_np=None,
        sv_fpn='./pred.drutai',
    )
    print(p.predict())
