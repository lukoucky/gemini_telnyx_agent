import os
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
import dotenv

from handlers.webhook_handler import WebhookHandler
from handlers.agent_handler import get_websocket_handler

# Load environment variables first
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Reduce external logging noise
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)

# Create FastAPI app
app = FastAPI(title="Telnyx Voice Agent", version="1.0.0")

# Configuration
NGROK_URL = os.getenv("NGROK_URL")
STREAM_URL = f"wss://{NGROK_URL}/audio"
TEST_AUDIO_FILE = "test_audio.mp3"  # Optional: Add an MP3 file to play

# Initialize handlers (after environment is loaded)
webhook_handler = WebhookHandler(STREAM_URL)

# Get agent type from environment (simple, gemini, legacy)
AGENT_TYPE = os.getenv("VOICE_AGENT_TYPE", "simple")
websocket_handler = get_websocket_handler(AGENT_TYPE)

# Ensure audio logs directory exists
os.makedirs("audio_logs", exist_ok=True)

logger.info(f"🚀 Telnyx Voice Agent starting...")
logger.info(f"📞 Stream URL: {STREAM_URL}")
logger.info(f"🎵 Test audio: {TEST_AUDIO_FILE if os.path.exists(TEST_AUDIO_FILE) else 'None (will use tones)'}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Telnyx Voice Agent",
        "timestamp": datetime.now().isoformat(),
        "stream_url": STREAM_URL
    }


@app.get("/test")
async def test_endpoint():
    """Test endpoint for connectivity verification"""
    return {
        "status": "ok",
        "message": "Telnyx Voice Agent is running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "webhook": "/webhook",
            "websocket": "/audio",
            "health": "/"
        }
    }


@app.post("/webhook")
async def webhook_endpoint(request: Request):
    """Handle Telnyx webhook events"""
    logger.info("📨 Webhook event received")
    return await webhook_handler.handle_webhook(request)


@app.websocket("/audio")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket audio streaming"""
    logger.info("🔌 WebSocket connection requested")
    await websocket_handler.handle_audio_websocket(websocket)


if __name__ == "__main__":
    import uvicorn
    
    logger.info("=" * 50)
    logger.info("🎙️  TELNYX VOICE AGENT - WITH AI AGENTS")
    logger.info("=" * 50)
    logger.info("🤖 Agent Integration:")
    logger.info(f"   • Agent Type: {AGENT_TYPE.upper()}")
    logger.info("   • Extensible agent architecture")
    logger.info("   • Real-time audio conversion (PCMU ↔ PCM)")
    logger.info("   • Bidirectional conversation support")
    logger.info("=" * 50)
    logger.info("🔧 Audio fixes applied:")
    logger.info("   • Direct ulaw2lin with width=2 for incoming audio")
    logger.info("   • Skip problematic lin2lin double-conversion")
    logger.info("   • Filter out tiny RTP payload artifacts")
    logger.info("   • Improved WebSocket timing for better capture")
    logger.info("=" * 50)
    logger.info("📋 Configuration:")
    logger.info(f"   • Stream URL: {STREAM_URL}")
    logger.info(f"   • Audio logs: ./audio_logs/")
    logger.info(f"   • Test audio: {TEST_AUDIO_FILE if os.path.exists(TEST_AUDIO_FILE) else 'Tones only'}")
    logger.info("=" * 50)
    logger.info("🌐 Starting server on 0.0.0.0:8080")
    logger.info("🔧 Configured for ngrok compatibility")
    logger.info("=" * 50)
    
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")