const { AccessToken } = require("livekit-server-sdk");

const LIVEKIT_URL = process.env.LIVEKIT_URL;
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY || process.env.API_KEY;
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET || process.env.API_SECRET;

const setCors = (res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
};

module.exports = async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });

  try {
    if (!LIVEKIT_URL || !LIVEKIT_API_KEY || !LIVEKIT_API_SECRET) {
      return res.status(500).json({
        error: "LiveKit env vars missing. Set LIVEKIT_URL and LIVEKIT_API_KEY/LIVEKIT_API_SECRET in Vercel.",
      });
    }

    const identity = "user-" + Math.floor(Math.random() * 10000);
    const roomName = "room-" + identity;

    const at = new AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET, { identity });
    at.addGrant({ roomJoin: true, room: roomName, canPublish: true, canSubscribe: true });

    const token = await at.toJwt();
    console.log("✅ Token generated for:", identity, "room:", roomName);
    return res.status(200).json({ token, roomName });

  } catch (err) {
    console.error("❌ Token generation failed:", err.message, err.stack);
    return res.status(500).json({ error: err.message });
  }
};
