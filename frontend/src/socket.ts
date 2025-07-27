import { io } from 'socket.io-client';

const socketUrl = 'http://0.0.0.0:3000';
export const socket = io(socketUrl, {
  transports: ['websocket'],
});
