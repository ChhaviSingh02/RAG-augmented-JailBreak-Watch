"""
Detection Pipeline — Orchestrates embedding, RAG, classifier, and agent
"""

import torch
from sentence_transformers import SentenceTransformer
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from qdrant_client import QdrantClient
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))
from pipeline.agent import JailbreakAgent

class JailbreakDetector:
    """Complete detection pipeline"""
    
    def __init__(self):
        print("Initializing Jailbreak Detector...")
        
        # Load embedding model
        print("  Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Connect to Qdrant
        print("  Connecting to Qdrant...")
        self.qdrant = QdrantClient(host="localhost", port=6333)
        self.collection_name = "jailbreak_patterns"
        
        # Load classifier
        print("  Loading classifier...")
        classifier_path = "models/classifier"
        if not Path(classifier_path).exists():
            raise FileNotFoundError(
                f"Classifier not found at {classifier_path}. "
                "Run 'python pipeline/classifier.py' first."
            )
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(classifier_path)
        self.classifier = DistilBertForSequenceClassification.from_pretrained(classifier_path)
        self.classifier.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.classifier.to(self.device)
        
        # Initialize agent
        print("  Initializing Ollama agent...")
        self.agent = JailbreakAgent(model="gpt-oss:120b-cloud")
        
        print("✓ Detector initialized and ready")
    
    def get_similarity_and_matches(self, prompt_text, top_k=5):
        """Embed prompt and search Qdrant for similar jailbreaks"""
        # Embed the prompt
        embedding = self.embedding_model.encode(prompt_text).tolist()
        
        # Search Qdrant
        results = self.qdrant.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=top_k
        )
        
        # Get highest similarity score
        similarity_score = results[0].score if results else 0.0
        
        # Format matches
        matches = [
            {
                "text": hit.payload['text'],
                "score": hit.score,
                "attack_type": hit.payload['attack_type']
            }
            for hit in results
        ]
        
        return similarity_score, matches
    
    def classify_prompt(self, prompt_text):
        """Run classifier to get label and confidence"""
        # Tokenize
        encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.classifier(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = probs[0][prediction].item()
        
        return prediction, confidence
    
    def detect(self, prompt_text):
        """
        Run full detection pipeline
        
        Returns:
            dict with all detection signals and agent decision
        """
        # 1. RAG similarity search
        similarity_score, top_matches = self.get_similarity_and_matches(prompt_text)
        
        # 2. Classifier
        classifier_label, classifier_confidence = self.classify_prompt(prompt_text)
        
        # 3. Agent decision
        agent_result = self.agent.analyze(
            prompt_text=prompt_text,
            similarity_score=similarity_score,
            top_matches=top_matches,
            classifier_label=classifier_label,
            classifier_confidence=classifier_confidence
        )
        
        # 4. Combine all signals
        result = {
            "text": prompt_text,
            "similarity_score": float(similarity_score),
            "top_matches": top_matches[:5],
            "classifier_label": int(classifier_label),
            "classifier_confidence": float(classifier_confidence),
            "agent_action": agent_result['action'],
            "agent_confidence": agent_result['confidence'],
            "attack_vector": agent_result['attack_vector'],
            "escalation": agent_result['escalation'],
            "reasoning": agent_result['reasoning'],
            "is_safe": classifier_label == 0 and similarity_score < 0.5
        }
        
        return result

# Singleton instance
_detector_instance = None

def get_detector():
    """Get or create detector singleton"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = JailbreakDetector()
    return _detector_instance