"""
FastAPI Server — REST API + WebSocket for real-time dashboard
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
from datetime import datetime
import uuid

from detector import get_detector
from database import (
    init_database, 
    store_prompt_analysis, 
    store_human_decision,
    get_audit_trail,
    get_statistics
)

app = FastAPI(title="Jailbreak Detection API")

# CORS for React dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

# Pydantic models
class AnalyzeRequest(BaseModel):
    prompt: str

class HumanDecisionRequest(BaseModel):
    prompt_id: str
    action: str
    note: Optional[str] = ""
    severity: Optional[int] = 3

# Initialize on startup
@app.on_event("startup")
async def startup():
    print("Starting Jailbreak Detection API...")
    await init_database()
    # Warm up the detector
    _ = get_detector()
    print("✓ API ready")

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Analyze endpoint
@app.post("/analyze")
async def analyze_prompt(request: AnalyzeRequest):
    """Analyze a prompt through the full detection pipeline"""
    try:
        detector = get_detector()
        
        # Run detection
        result = detector.detect(request.prompt)
        
        # Generate unique ID
        prompt_id = str(uuid.uuid4())[:8]
        result['prompt_id'] = prompt_id
        result['timestamp'] = datetime.now().isoformat()
        
        # Store in database
        await store_prompt_analysis(prompt_id, result)
        
        # Broadcast to WebSocket clients
        await manager.broadcast({
            "type": "new_detection",
            "data": result
        })
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Human decision endpoint
@app.post("/human-decision")
async def human_decision(request: HumanDecisionRequest):
    """Record human oversight decision"""
    try:
        await store_human_decision(
            request.prompt_id,
            request.action,
            request.note,
            request.severity
        )
        
        # Broadcast update
        await manager.broadcast({
            "type": "human_decision",
            "data": {
                "prompt_id": request.prompt_id,
                "action": request.action,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return {"status": "success", "prompt_id": request.prompt_id}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Audit trail endpoint
@app.get("/audit-trail")
async def audit_trail(limit: int = 100):
    """Get recent audit trail entries"""
    try:
        entries = await get_audit_trail(limit)
        return {"entries": entries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Statistics endpoint
@app.get("/statistics")
async def statistics():
    """Get dashboard statistics"""
    try:
        stats = await get_statistics()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for live feed
@app.websocket("/ws/feed")
async def websocket_feed(websocket: WebSocket):
    """WebSocket for real-time dashboard updates"""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)