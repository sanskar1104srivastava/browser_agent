import { AgentDispatchClient } from "livekit-server-sdk";

const LIVEKIT_URL = process.env.LIVEKIT_URL;
const LIVEKIT_API_KEY = process.env.LIVEKIT_API_KEY || process.env.API_KEY;
const LIVEKIT_API_SECRET = process.env.LIVEKIT_API_SECRET || process.env.API_SECRET;

const setCors = (res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "*");
  res.setHeader("Access-Control-Allow-Methods", "GET,POST,OPTIONS");
};

export default async function handler(req, res) {
  setCors(res);
  if (req.method === "OPTIONS") return res.status(200).end();
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  try {
    if (!LIVEKIT_URL || !LIVEKIT_API_KEY || !LIVEKIT_API_SECRET) {
      return res.status(500).json({
        error: "LiveKit env vars missing. Set LIVEKIT_URL and LIVEKIT_API_KEY/LIVEKIT_API_SECRET in Vercel.",
      });
    }

    const { roomName } = req.body;
    if (!roomName) return res.status(400).json({ error: "roomName is required" });

    console.log("📡 Dispatching agent to room:", roomName);
    const agentDispatch = new AgentDispatchClient(LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET);
    await agentDispatch.createDispatch(roomName, "voice-bot");
    console.log("✅ Agent dispatched to:", roomName);
    return res.status(200).json({ success: true });

  } catch (err) {
    console.error("❌ Agent dispatch failed:", err.message, err.stack);
    return res.status(500).json({ error: err.message });
  }
}
