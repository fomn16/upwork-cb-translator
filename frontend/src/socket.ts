import { io } from 'socket.io-client';

const socketUrl = 'http://127.0.0.1:3000';
export const socket = io(socketUrl, {
  transports: ['websocket'],
});
