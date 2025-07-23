import os
import logging
from datetime import datetime
from fastapi import FastAPI, WebSocket, Request
from fastapi.responses import JSONResponse
import dotenv

from handlers.webhook_handler import WebhookHandler
from handlers.websocket_handler import WebSocketHandler

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
NGROK_URL = "c8d00a2a4259.ngrok-free.app"  # Update this with your ngrok URL
STREAM_URL = f"wss://{NGROK_URL}/audio"
TEST_AUDIO_FILE = "test_audio.mp3"  # Optional: Add an MP3 file to play

# Initialize handlers (after environment is loaded)
webhook_handler = WebhookHandler(STREAM_URL)
websocket_handler = WebSocketHandler(TEST_AUDIO_FILE)

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
    logger.info("🎙️  TELNYX VOICE AGENT")
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