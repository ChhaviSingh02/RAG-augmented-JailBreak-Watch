"""
Process unified_bench.json and unified_bench.csv into normalized format
"""

import json
import pandas as pd
from pathlib import Path
import re

# Ensure directories exist
Path("data/processed").mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Remove excessive whitespace, normalize formatting"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def process_unified_data():
    """
    Process your existing unified_bench files
    Normalize into research-ready schema
    """
    print("\n[1/2] Loading unified_bench.csv...")
    
    # Load the CSV
    df = pd.read_csv("data/raw/unified_bench.csv")
    print(f"  Loaded {len(df)} rows from unified_bench.csv")
    
    # Normalize into standard schema
    normalized = []
    
    for idx, row in df.iterrows():
        # Extract metadata tags if present
        tags = row.get('metadata_tags', '')
        if pd.isna(tags):
            tags = ''
        
        # Determine attack type from tags
        attack_type = "jailbreak"
        if "Harassment" in tags:
            attack_type = "harassment"
        elif "Defamation" in tags:
            attack_type = "defamation"
        elif "Discrimination" in tags:
            attack_type = "discrimination"
        elif "Violence" in tags:
            attack_type = "violence"
        
        normalized.append({
            "id": row['id'],
            "text": clean_text(row['prompt']),
            "label": 1,  # All are jailbreak attempts
            "attack_type": attack_type,
            "category": row.get('category', 'jailbreak'),
            "source": row.get('benchmark', 'JailbreakBench'),
            "response": clean_text(row.get('response', '')) if pd.notna(row.get('response')) else '',
            "original_label": row.get('label', 'success')
        })
    
    jailbreak_df = pd.DataFrame(normalized)
    print(f"✓ Processed {len(jailbreak_df)} jailbreak prompts")
    
    return jailbreak_df

def generate_safe_prompts(n=300):
    """Generate safe prompts as negative examples"""
    print(f"\n[2/2] Generating {n} safe prompts...")
    
    safe_templates = [
        "What's the capital of {}?",
        "Explain {} in simple terms",
        "Help me write a {} function in Python",
        "Summarize the history of {}",
        "What are the benefits of {}?",
        "How does {} work?",
        "Write a poem about {}",
        "What's the best way to learn {}?",
        "Compare {} and {}",
        "Debug this code: {}",
        "Translate '{}' to Spanish",
        "What are common uses of {}?",
        "Recommend books about {}",
        "Explain the difference between {} and {}",
        "Write a tutorial on {}",
        "What is the etymology of {}?",
        "How can I improve my {} skills?",
        "Create a study guide for {}",
        "What are the main principles of {}?",
        "Describe the process of {}"
    ]
    
    topics = [
        "France", "quantum physics", "sorting", "the Roman Empire", "meditation",
        "photosynthesis", "autumn leaves", "Spanish", "REST APIs", "machine learning",
        "JavaScript", "renewable energy", "Shakespeare", "democracy", "blockchain",
        "neural networks", "cooking", "exercise", "time management", "creativity",
        "climate change", "psychology", "economics", "art history", "music theory",
        "biology", "chemistry", "geography", "literature", "mathematics"
    ]
    
    prompts = []
    for idx in range(n):
        template = safe_templates[idx % len(safe_templates)]
        count = template.count('{}')
        
        if count == 2:
            topic1 = topics[idx % len(topics)]
            topic2 = topics[(idx + 1) % len(topics)]
            text = template.format(topic1, topic2)
        elif count == 1:
            topic = topics[idx % len(topics)]
            text = template.format(topic)
        else:
            text = template
        
        prompts.append({
            "id": f"safe_{idx}",
            "text": text,
            "label": 0,  # Safe
            "attack_type": "none",
            "category": "safe",
            "source": "Generated",
            "response": "",
            "original_label": ""
        })
    
    return pd.DataFrame(prompts)

def merge_and_split():
    """Merge jailbreak and safe prompts, create train/test split"""
    print("\n[MERGE] Combining datasets...")
    
    # Load jailbreak data
    jailbreak_df = process_unified_data()
    
    # Generate safe prompts
    safe_df = generate_safe_prompts(300)
    
    # Merge
    merged = pd.concat([jailbreak_df, safe_df], ignore_index=True)
    
    # Remove duplicates
    merged = merged.drop_duplicates(subset=['text'], keep='first')
    
    # Shuffle
    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Stats
    n_jailbreak = (merged['label'] == 1).sum()
    n_safe = (merged['label'] == 0).sum()
    
    print(f"\n[STATS]")
    print(f"  Total prompts: {len(merged)}")
    print(f"  Jailbreak/Harmful: {n_jailbreak} ({n_jailbreak/len(merged)*100:.1f}%)")
    print(f"  Safe: {n_safe} ({n_safe/len(merged)*100:.1f}%)")
    print(f"  Unique attack types: {merged['attack_type'].nunique()}")
    print(f"  Attack type breakdown:")
    for attack_type, count in merged['attack_type'].value_counts().items():
        print(f"    - {attack_type}: {count}")
    
    # Split 80/20
    split_idx = int(len(merged) * 0.8)
    train = merged[:split_idx]
    test = merged[split_idx:]
    
    # Save
    merged.to_csv("data/processed/all_prompts.csv", index=False)
    train.to_csv("data/processed/train.csv", index=False)
    test.to_csv("data/processed/test.csv", index=False)
    
    print(f"\n[SAVED]")
    print(f"  data/processed/all_prompts.csv ({len(merged)} rows)")
    print(f"  data/processed/train.csv ({len(train)} rows)")
    print(f"  data/processed/test.csv ({len(test)} rows)")
    
    # Save metadata for research paper
    metadata = {
        "total_prompts": len(merged),
        "jailbreak_count": int(n_jailbreak),
        "safe_count": int(n_safe),
        "train_size": len(train),
        "test_size": len(test),
        "sources": merged['source'].value_counts().to_dict(),
        "attack_types": merged['attack_type'].value_counts().to_dict(),
        "original_dataset": "JailbreakBench unified_bench"
    }
    
    with open("data/processed/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"  data/processed/metadata.json")
    
    # Save a sample for verification
    sample = merged.head(20)
    sample.to_csv("data/processed/sample.csv", index=False)
    print(f"  data/processed/sample.csv (20 rows for inspection)")
    
    return merged, train, test

if __name__ == "__main__":
    print("=" * 60)
    print("JAILBREAK DETECTION — DATA PROCESSING PIPELINE")
    print("=" * 60)
    
    # Process and merge
    all_data, train_data, test_data = merge_and_split()
    
    print("\n" + "=" * 60)
    print("✓ DATA PROCESSING COMPLETE")
    print("=" * 60)
    print("\nYour processed data is ready for:")
    print("  1. Embedding → Qdrant vector DB")
    print("  2. Classifier training")
    print("  3. Real-time detection pipeline")
    print("\nNext step:")
    print("  python pipeline/embed.py")