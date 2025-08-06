# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

You are never under pressure to give me quick answer! I 100% prefer to have correct answer that takes a long time rather then quick one in few seconds. 

Make sure you use this file every time you give me any answer or do a code change!! Please use this file and follow what is written here!! Please please do it!! You make me go crazy when I have to keep repeating you what is written here into prompts. JUST USE THIS FILE!!!!

## Project Overview

This is a Telnyx voice agent application that creates a WebSocket-based voice streaming service. This is just a simple POC to test Telnyx. Endgoal is to integrate realtime AI voice agent and talk to him over the real phone - using Telnyx as a service for phone calls. The system handles incoming phone calls, establishes WebSocket connections for real-time audio streaming, and can integrate with AI voice processing services.

### ⚠️ IMPORTANT: Development Guidelines

- Most important: YOU NEVER LIE TO ME. Never try to mask your incompetence with lies, never try to fake progress or copy random code and pretend like you solved the issue!
- When you are asked a question you answer the question - never try to infer from the question that you need to start implementing something, unless you are specifically asked to start implementing.
- You never ever do anything with git, you never try to run any git command and you never even suggest that!!
- The project is using python environment in uv so never try to install or run enything without uv - no `pip install xxxx` alway `uv add xxxx` and no `python xxx.py` but always `uv run xxx.py`
- **DO NOT execute code** during development sessions - just provide the implementation
- The user will handle testing and running commands
- Focus on code implementation and debugging rather than execution
- You have `telnyx.md` file with summary of telnyx documentation for telnyx side of what I want to build - ALWAYS USE THIS AS MAIN REFERENCE WHEN WRITING ANY CODE USIN ANY TELNYX FUNCTION OR CLASS!!! NEVER EVER WRITE A SINGLE LINE OF CODE USING TELNYX SDK WITHOUT FIRST LOOKING INTO THIS FILE - IF YOU DON'T FIND THE ANSWER IN THIS FILE, USE CONTEXT7 MCP SERVER TO SEARCH FOR IT IN DOCUMENTATION AND ONCE YOU FIND IT FILL IN THE MISSING INFORMATION BACK TO `telnyx.md`
- You have `google_agents.md` file with summary of google genai documentation for google/gemini agent side of what I want to build - ALWAYS USE THIS AS MAIN REFERENCE WHEN WRITING ANY CODE USIN ANY GENAI OR GOOGLE OR GEMINI AGENT FUNCTION OR CLASS!!! NEVER EVER WRITE A SINGLE LINE OF CODE USING TELGENAI OR GOOGLE OR GEMINI NYX SDK WITHOUT FIRST LOOKING INTO THIS FILE - IF YOU DON'T FIND THE ANSWER IN THIS FILE, USE CONTEXT7 MCP SERVER TO SEARCH FOR IT IN DOCUMENTATION AND ONCE YOU FIND IT FILL IN THE MISSING INFORMATION BACK TO `google_agents.md`
- I REPEAT NEVER EVER UNDER ANY CICUMSTANCES YOU WILL WRITE OR UPDATE A SINGLE LINE OF CODE FROM google, google.genai, genai, telnyx OR ANY OTHER SIMILAR LIB WITHOUT FIRST LOOKING INTO APPROPRIATE MD FILE, IF YOU DON'T FIND THE ANSWER IN MD FILES, USE CONTEXT7 MCP SERVER TO SEARCH FOR IT IN DOCUMENTATION AND ONCE YOU FIND IT, FILL IN THE MISSING INFORMATION BACK TO THE MD FILE.
- Never put imports in middle of file or after the condition statements - always put imports to the top of the files using python best practices

Environment variables in `.env` file are:
- `TELNYX_API_KEY`: Your Telnyx API key for phone service integration
- `GEMINI_API_KEY`: Your Gemini (google) API key for agent
