"""
Prompt Simulator — Replays test prompts to feed the live dashboard
"""

import requests
import pandas as pd
import time
import random
from pathlib import Path

API_URL = "http://localhost:8000"

def simulate_live_feed():
    """Simulate live prompts from test dataset"""
    print("=" * 60)
    print("PROMPT SIMULATOR — Live Feed")
    print("=" * 60)
    
    # Load test data
    test_path = "data/processed/test.csv"
    if not Path(test_path).exists():
        print(f"✗ Test data not found at {test_path}")
        return
    
    df = pd.read_csv(test_path)
    print(f"Loaded {len(df)} test prompts")
    print(f"Starting simulation... (Ctrl+C to stop)\n")
    
    # Shuffle for variety
    df = df.sample(frac=1).reset_index(drop=True)
    
    try:
        for idx, row in df.iterrows():
            prompt_text = row['text']
            actual_label = row['label']
            
            print(f"[{idx+1}/{len(df)}] Sending: {prompt_text[:60]}...")
            
            try:
                response = requests.post(
                    f"{API_URL}/analyze",
                    json={"prompt": prompt_text},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"  → Action: {result['agent_action']}")
                    print(f"  → Similarity: {result['similarity_score']:.3f}")
                    print(f"  → Classifier: {'Jailbreak' if result['classifier_label']==1 else 'Safe'}")
                else:
                    print(f"  ✗ Error: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                print(f"  ✗ Cannot connect to API at {API_URL}")
                print(f"  Make sure server is running: python api/main.py")
                break
            except Exception as e:
                print(f"  ✗ Error: {e}")
            
            # Random delay between prompts (2-5 seconds)
            delay = random.uniform(2, 5)
            time.sleep(delay)
    
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user")
    
    print("\n" + "=" * 60)
    print("✓ SIMULATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    simulate_live_feed()