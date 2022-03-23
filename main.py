import traceback
import numpy as np
from fastapi import Request
from infer import encode, sim, sim_adjust
from app import app


@app.post('/api/encode')
async def api_encode(request: Request):
    """
    time curl -XPOST http://localhost:8000/api/encode \
        -H 'Content-Type: applicaton/json' \
        -d '{"text": "你好啊"}'
    """
    data = await request.json()
    if 'text' not in data or not isinstance(data['text'], str):
        return {
            'ok': False,
            'error': 'Invalid text in post data',
        }
    try:
        vec = encode(data['text'])
        return {
            'ok': True,
            'data': vec,
        }
    except Exception:
        return {
            'ok': False,
            'error': traceback.format_exc(),
        }


@app.get('/api/encode/{text}')
async def api_encode(text: str, request: Request):
    """
    time curl http://localhost:8000/api/encode/hello
    """
    if not isinstance(text, str) or not text.strip():
        return {
            'ok': False,
            'error': 'Invalid text in query',
        }
    try:
        vec = encode(text)
        return {
            'ok': True,
            'data': vec,
        }
    except Exception:
        return {
            'ok': False,
            'error': traceback.format_exc(),
        }


@app.post('/api/sim')
async def api_sim(request: Request):
    """
    time curl -XPOST http://localhost:8000/api/sim \
        -H 'Content-Type: applicaton/json' \
        -d '{"a": "你好啊", "b": "你好"}'
    """
    data = await request.json()
    if 'a' not in data or not isinstance(data['a'], str):
        return {
            'ok': False,
            'error': 'Invalid a in post data',
        }
    if 'b' not in data or not isinstance(data['b'], str):
        return {
            'ok': False,
            'error': 'Invalid b in post data',
        }
    try:
        ret = sim(data['a'], data['b'])
        ret = float(ret)
        return {
            'ok': True,
            'data': ret,
        }
    except Exception:
        return {
            'ok': False,
            'error': traceback.format_exc(),
        }


@app.post('/api/sim_adjust')
async def api_sim(request: Request):
    """
    time curl -XPOST http://localhost:8000/api/sim_adjust \
        -H 'Content-Type: applicaton/json' \
        -d '{"a": "你好啊", "b": "你好"}'
    """
    data = await request.json()
    if 'a' not in data or not isinstance(data['a'], str):
        return {
            'ok': False,
            'error': 'Invalid a in post data',
        }
    if 'b' not in data or not isinstance(data['b'], str):
        return {
            'ok': False,
            'error': 'Invalid b in post data',
        }
    try:
        ret = sim_adjust(data['a'], data['b'])
        ret = float(ret)
        return {
            'ok': True,
            'data': ret,
        }
    except Exception:
        return {
            'ok': False,
            'error': traceback.format_exc(),
        }


@app.get('/')
async def hello():
    return {
        'hello': 'world',
    }
