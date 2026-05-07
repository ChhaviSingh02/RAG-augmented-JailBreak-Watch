"""
Database layer — SQLite for audit trail and human feedback
"""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "audit.db"

async def init_database():
    """Initialize SQLite database with required tables"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Prompts table — stores all analyzed prompts
        await db.execute("""
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT UNIQUE,
                text TEXT NOT NULL,
                similarity_score REAL,
                classifier_label INTEGER,
                classifier_confidence REAL,
                agent_action TEXT,
                agent_confidence INTEGER,
                attack_vector TEXT,
                escalation TEXT,
                reasoning TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Audit trail — stores human decisions and feedback
        await db.execute("""
            CREATE TABLE IF NOT EXISTS audit_trail (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                action TEXT NOT NULL,
                note TEXT,
                severity INTEGER,
                human_override BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id)
            )
        """)
        
        # Top matches cache — stores RAG retrieval results
        await db.execute("""
            CREATE TABLE IF NOT EXISTS rag_matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_id TEXT,
                match_text TEXT,
                match_score REAL,
                match_rank INTEGER,
                FOREIGN KEY (prompt_id) REFERENCES prompts(prompt_id)
            )
        """)
        
        await db.commit()
        print(f"✓ Database initialized: {DB_PATH}")

async def store_prompt_analysis(prompt_id, data):
    """Store a complete prompt analysis"""
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO prompts (
                prompt_id, text, similarity_score, classifier_label, 
                classifier_confidence, agent_action, agent_confidence,
                attack_vector, escalation, reasoning, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            prompt_id,
            data['text'],
            data['similarity_score'],
            data['classifier_label'],
            data['classifier_confidence'],
            data['agent_action'],
            data['agent_confidence'],
            data['attack_vector'],
            data['escalation'],
            data['reasoning'],
            'pending'
        ))
        
        # Store top matches
        for rank, match in enumerate(data.get('top_matches', []), 1):
            await db.execute("""
                INSERT INTO rag_matches (prompt_id, match_text, match_score, match_rank)
                VALUES (?, ?, ?, ?)
            """, (prompt_id, match['text'], match['score'], rank))
        
        await db.commit()

async def store_human_decision(prompt_id, action, note="", severity=3):
    """Store human oversight decision"""
    async with aiosqlite.connect(DB_PATH) as db:
        # Insert audit entry
        await db.execute("""
            INSERT INTO audit_trail (prompt_id, action, note, severity, human_override)
            VALUES (?, ?, ?, ?, 1)
        """, (prompt_id, action, note, severity))
        
        # Update prompt status
        await db.execute("""
            UPDATE prompts SET status = ? WHERE prompt_id = ?
        """, (action, prompt_id))
        
        await db.commit()

async def get_audit_trail(limit=100):
    """Retrieve recent audit trail entries"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("""
            SELECT 
                a.id,
                a.prompt_id,
                p.text as prompt_text,
                a.action,
                p.similarity_score as score,
                a.note,
                a.severity,
                a.created_at as ts
            FROM audit_trail a
            JOIN prompts p ON a.prompt_id = p.prompt_id
            ORDER BY a.created_at DESC
            LIMIT ?
        """, (limit,)) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

async def get_statistics():
    """Get system statistics for dashboard"""
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        
        # Total prompts
        async with db.execute("SELECT COUNT(*) as count FROM prompts") as cursor:
            total = (await cursor.fetchone())['count']
        
        # Jailbreak count
        async with db.execute("""
            SELECT COUNT(*) as count FROM prompts WHERE classifier_label = 1
        """) as cursor:
            jailbreaks = (await cursor.fetchone())['count']
        
        # Blocked count
        async with db.execute("""
            SELECT COUNT(*) as count FROM prompts WHERE status = 'Block'
        """) as cursor:
            blocked = (await cursor.fetchone())['count']
        
        # Pending count
        async with db.execute("""
            SELECT COUNT(*) as count FROM prompts WHERE status = 'pending'
        """) as cursor:
            pending = (await cursor.fetchone())['count']
        
        return {
            "total": total,
            "jailbreaks": jailbreaks,
            "blocked": blocked,
            "pending": pending
        }