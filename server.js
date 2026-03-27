const express = require('express');
const { WebSocketServer } = require('ws');
const http = require('http');

const app = express();
const server = http.createServer(app);
const wss = new WebSocketServer({ server });

// NOTE to panel: This is Layer 4 Communication handler (WebSocket TLS 1.3 to be configured)
// It handles JSON payloads: event, confidence, timestamp without transmitting video
wss.on('connection', function connection(ws) {
  console.log("New client connected to FallGuard Hub");
  
  ws.on('message', function incoming(message) {
    console.log('received payload: %s', message);
    
    // Broadcast back to all connected dashboard clients
    wss.clients.forEach(function each(client) {
      if (client !== ws && client.readyState === 1) {
        client.send(message.toString());
      }
    });
  });

  ws.send(JSON.stringify({ status: 'Connected to FallGuard Server' }));
});

const PORT = process.env.PORT || 8080;
server.listen(PORT, () => {
  console.log(`FallGuard Integration Server started on port ${PORT}`);
});
