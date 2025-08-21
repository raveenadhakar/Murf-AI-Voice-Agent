async def stream_llm_response(user_input: str, websocket: WebSocket):
    """
    Stream LLM response to WebSocket client and accumulate for console output
    """
    if not model:
        logger.error("❌ [LLM]: Model not available")
        await websocket.send_text(json.dumps({
            "type": "llm_error",
            "message": "LLM service unavailable"
        }))
        return
    
    try:
        logger.info(f"🤖 [LLM]: Processing user input: '{user_input}'")
        
        # Create a conversational prompt
        prompt = f"""You are a helpful AI assistant. Please provide a natural, conversational response to the user's message. Keep your response concise but engaging.

User: {user_input}Assist
ant: Please respond naturally and helpfully."""
        
        # Send LLM start notification
        await websocket.send_text(json.dumps({
            "type": "llm_start",
            "message": "AI is thinking..."
        }))
        
        # Generate streaming response
        response_stream = model.generate_content(prompt, stream=True)
        accumulated_response = ""
        chunk_count = 0
        
        logger.info("🚀 [LLM]: Starting streaming response...")
        
        for chunk in response_stream:
            if chunk.text:
                chunk_count += 1
                accumulated_response += chunk.text
                
                # Log each chunk to console
                logger.info(f"🤖 [LLM CHUNK #{chunk_count}]: {chunk.text}")
                
                # Send chunk to client
                await websocket.send_text(json.dumps({
                    "type": "llm_chunk",
                    "chunk": chunk.text,
                    "chunk_number": chunk_count
                }))
        
        # Send completion notification
        await websocket.send_text(json.dumps({
            "type": "llm_complete",
            "full_response": accumulated_response,
            "total_chunks": chunk_count
        }))
        
        # Log complete response to console
        logger.info(f"✅ [LLM COMPLETE]: {accumulated_response}")
        logger.info(f"📊 [LLM STATS]: {chunk_count} chunks, {len(accumulated_response)} characters")
        
    except Exception as e:
        logger.error(f"❌ [LLM]: Error during streaming: {e}")
        await websocket.send_text(json.dumps({
            "type": "llm_error",
            "message": f"LLM error: {str(e)}"
        }))