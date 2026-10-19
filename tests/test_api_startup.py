import asyncio
import time

from riskengine import api


def test_startup_waits_for_the_model_warm_up(monkeypatch):
    done = []

    def slow_warm_up():
        time.sleep(0.2)
        done.append(True)

    monkeypatch.setattr(api, "warm_up", slow_warm_up)

    async def serve():
        async with api.lifespan(api.app):
            return list(done)  # what a first request would see

    assert asyncio.run(serve()) == [True]
