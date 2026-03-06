import os
import asyncio
from pathlib import Path

import pandas as pd

from drutai.util.Feature import fetch_from_data
from drutai.util.Model import Model
from drutai.web.schemas import PredictRequest, PredictResponse, Prediction, MethodInfo


# method name → model subpath
_METHOD_MAP = {
    'AlexNet':      'alexnet/alexnet',
    'BiRNN':        'birnn/birnn',
    'RNN':          'rnn/rnn',
    'Seq2Seq':      'seq2seq/seq2seq',
    'ResNet50':     'resnet50/resnet50',
    'CNN':          'cnn',
    'ConvMixer64':  'convmixer64',
    'DSConv':       'dsconv',
    'LSTMCNN':      'lstmcnn',
    'MobileNet':    'mobilenetv2',
    'ResNet18':     'resnet_prea18_tf2',
    'SEResNet':     'scaresnet',
}


def _data_dir() -> Path:
    """Resolve the data/ directory relative to the project root."""
    return Path(os.environ.get('DRUTAI_DATA_DIR', 'data'))


def list_methods() -> list[MethodInfo]:
    data_dir = _data_dir()
    results = []
    for name, subpath in _METHOD_MAP.items():
        model_path = data_dir / subpath
        available = model_path.with_suffix('.onnx').exists()
        results.append(MethodInfo(name=name, available=available))
    return results


def _run_prediction(req: PredictRequest) -> PredictResponse:
    """Blocking prediction (runs inside executor)."""
    method = req.method
    if method not in _METHOD_MAP:
        raise ValueError(f"Unknown method: {method}. Available: {list(_METHOD_MAP.keys())}")

    data_dir = _data_dir()
    model_fp = str(data_dir / _METHOD_MAP[method])

    # Build DataFrames from request
    df_br = pd.DataFrame([{'sm': st.sm, 'target': st.target} for st in req.sm_targets])
    df_smile = pd.DataFrame([{'sm': ss.sm, 'smile': ss.smile} for ss in req.sm_smiles])
    fasta_sequences = {fe.target_id: fe.sequence for fe in req.fasta_entries}

    # Feature extraction (CPU-bound but not TF-dependent)
    mat_np = fetch_from_data(df_br, df_smile, fasta_sequences, verbose=False)

    # Inference (ONNX sessions are thread-safe — no lock needed)
    model = Model(
        mat_np=mat_np,
        method=method,
        model_fp=model_fp,
        sv_fpn=None,
        verbose=False,
    )
    df_pred = model.predict()

    # Combine predictions with input identifiers
    predictions = []
    for i in range(len(df_br)):
        predictions.append(Prediction(
            sm=int(df_br.loc[i, 'sm']),
            target=df_br.loc[i, 'target'],
            prob_inter=float(df_pred.loc[i, 'prob_inter']),
            pred_type=df_pred.loc[i, 'pred_type'],
        ))

    return PredictResponse(
        method=method,
        num_samples=len(predictions),
        predictions=predictions,
    )


async def predict(req: PredictRequest) -> PredictResponse:
    """Run prediction in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_prediction, req)
