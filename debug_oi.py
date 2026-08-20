
import asyncio
import aiohttp
import time
import pandas as pd

async def test_oi_chunked():
    symbol = "BTCUSDT"
    # End time = now, Start time = 60 days ago
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - (20 * 24 * 3600 * 1000)
    
    current_start = start_ms
    chunk_size = 7 * 24 * 3600 * 1000
    all_data = []

    print(f"Fetching OI for {symbol} from {start_ms} to {end_ms} (chunks of {chunk_size}ms)...")

    async with aiohttp.ClientSession() as session:
        while current_start < end_ms:
            current_end = min(current_start + chunk_size, end_ms)
            
            url = (
                f"https://fapi.binance.com/futures/data/openInterestHist"
                f"?symbol={symbol}&period=1h"
                f"&startTime={current_start}&endTime={current_end}"
                f"&limit=500"
            )
            print(f"Requesting: {current_start} -> {current_end}")
            
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if data:
                        all_data.extend(data)
                        print(f"  Passed! Got {len(data)} records.")
                    current_start = current_end + 1
                else:
                    print(f"  Failed: {resp.status} - {await resp.text()}")
                    current_start = current_end + 1
                    continue
            
            await asyncio.sleep(0.5)

    print(f"Total records fetched: {len(all_data)}")
    if len(all_data) > 0:
        print("First:", all_data[0]['timestamp'])
        print("Last :", all_data[-1]['timestamp'])

if __name__ == "__main__":
    asyncio.run(test_oi_chunked())
