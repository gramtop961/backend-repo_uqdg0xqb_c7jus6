import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
except Exception:
    analyzer = None

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SentimentIn(BaseModel):
    text: str

class VibeIn(BaseModel):
    mood: str
    weather_main: str | None = None
    weather_desc: str | None = None

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/api/weather")
def get_weather(city: str, api_key: str | None = None):
    """Fetch current weather for a city via OpenWeatherMap.
    Uses OPENWEATHER_API_KEY from env unless api_key query param provided.
    """
    key = api_key or os.getenv("OPENWEATHER_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENWEATHER_API_KEY not set on server; provide ?api_key=...")
    params = {
        "q": city,
        "appid": key,
        "units": "metric"
    }
    try:
        r = requests.get("https://api.openweathermap.org/data/2.5/weather", params=params, timeout=12)
        if r.status_code != 200:
            raise HTTPException(status_code=r.status_code, detail=r.text)
        data = r.json()
        # Normalize minimal shape for frontend
        result = {
            "city": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temp": data.get("main", {}).get("temp"),
            "feels_like": data.get("main", {}).get("feels_like"),
            "humidity": data.get("main", {}).get("humidity"),
            "wind": data.get("wind", {}).get("speed"),
            "weather_main": (data.get("weather") or [{}])[0].get("main"),
            "weather_desc": (data.get("weather") or [{}])[0].get("description"),
            "icon": (data.get("weather") or [{}])[0].get("icon"),
        }
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sentiment")
def analyze_sentiment(payload: SentimentIn):
    """Return simple sentiment for the provided text using VADER (lightweight)."""
    text = (payload.text or "").strip()
    if not text:
        return {"label": "NEUTRAL", "score": 0.0}
    if analyzer is None:
        # Fallback heuristic if analyzer isn't available for any reason
        lower = text.lower()
        positives = ["happy", "great", "awesome", "love", "good", "😊", ":)", "😀", "😄", "😎", "👍"]
        negatives = ["sad", "bad", "terrible", "hate", "angry", "😢", ":(", "😭", "😡", "😞", "👎"]
        p = sum(w in lower for w in positives)
        n = sum(w in lower for w in negatives)
        if p > n:
            return {"label": "POSITIVE", "score": min(1.0, 0.6 + 0.1 * (p - n))}
        if n > p:
            return {"label": "NEGATIVE", "score": min(1.0, 0.6 + 0.1 * (n - p))}
        return {"label": "NEUTRAL", "score": 0.5}
    vs = analyzer.polarity_scores(text)
    compound = vs.get("compound", 0.0)
    if compound >= 0.2:
        label = "POSITIVE"
    elif compound <= -0.2:
        label = "NEGATIVE"
    else:
        label = "NEUTRAL"
    return {"label": label, "score": round(compound, 4)}

@app.post("/api/vibe-note")
def vibe_note(payload: VibeIn):
    """Generate a short vibe note using simple rules based on mood and weather."""
    mood = (payload.mood or "").strip()
    wm = (payload.weather_main or "").lower()
    wd = (payload.weather_desc or "").lower()

    # Very small, deterministic generator without external AI
    base = "Today"
    if "rain" in wm or "drizzle" in wm:
        weather_phrase = "the rhythm of the rain sets a calm tempo"
    elif "cloud" in wm or "overcast" in wd:
        weather_phrase = "soft clouds invite reflection and slow moments"
    elif "clear" in wm:
        weather_phrase = "clear skies spark light and optimism"
    elif "snow" in wm:
        weather_phrase = "snow hushes the world into a gentle hush"
    elif "storm" in wm or "thunder" in wm:
        weather_phrase = "storms rumble, but you carry quiet courage"
    else:
        weather_phrase = "the air feels open to possibility"

    mood_lower = mood.lower()
    if any(k in mood_lower for k in ["happy", "great", "good", "joy", "excited", "😊", "😀", "😄", "😎", ":)"]):
        mood_phrase = "your energy feels bright and contagious"
    elif any(k in mood_lower for k in ["sad", "down", "tired", "low", "anxious", "😢", "😭", ":(", "😞"]):
        mood_phrase = "be gentle with yourself—small steps are still progress"
    elif any(k in mood_lower for k in ["angry", "frustrated", "mad", "😡"]):
        mood_phrase = "take a breath; let movement help shift the heat"
    else:
        mood_phrase = "you’re steady and present—keep following the simple good"

    note = f"{base}, {weather_phrase}; {mood_phrase}."
    return {"note": note}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
