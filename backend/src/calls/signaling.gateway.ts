import {
    SubscribeMessage,
    WebSocketGateway,
    OnGatewayInit,
} from '@nestjs/websockets';
import { Socket, Server } from 'socket.io';
import { MediasoupService } from './Mediasoup.service';
import {
    Producer,
    RtpCapabilities,
    WebRtcTransport,
    MediaKind,
    AppData,
    Consumer,
    PlainTransport,
    RtpCodecParameters
} from 'mediasoup/node/lib/types';
import { randomInt } from 'crypto';
import { getPort } from './port';

const rooms = new Map<
    string,
    {
        producers: Map<string, Producer>;
    }
>();

const userTransports = new Map<string, WebRtcTransport>();
const userProducers = new Map<string, Producer>();
const userConsumers = new Map<string, Consumer>();
const translationTransports = new Map<string, PlainTransport>();
const translationProducers = new Map<string, Producer>();
const translationConsumers = new Map<string, Consumer>();

let io: Server;

/* storing default configurations for bidirectional processing connections */
class BidirectionalConnectionSettings{
    enableVoiceClone: boolean
    processingServerUrl: string
    processingServerInitiateMethod: string
    successMessage: string
    errorMessage: string
    RTPCodecParameter:RtpCodecParameters[]
}
function getSettingsForKind(kind:'audio'|'video', codecs:RtpCodecParameters[]){
    let settings = new BidirectionalConnectionSettings()
    settings.enableVoiceClone = false
    settings.processingServerUrl = 'http://0.0.0.0:2002/'

    if(kind == 'audio'){
        settings.processingServerInitiateMethod = 'translation/initiate'
        settings.successMessage = '✅ Translation pipeline initiated:'
        settings.errorMessage = '❌ Error initiating translation pipeline:'
        settings.RTPCodecParameter = [
            {
                mimeType: 'audio/opus',
                payloadType:codecs[0].payloadType,
                clockRate:codecs[0].clockRate,
                channels:codecs[0].channels || 2
            }
        ]
    }
    else{
        settings.processingServerInitiateMethod = 'video/initiate'
        settings.successMessage = '✅ Video capture pipeline initiated:'
        settings.errorMessage = '❌ Error initiating video capture pipeline:'
        settings.RTPCodecParameter = [{
            mimeType:'video/VP8',
            payloadType:101,
            clockRate: 90000
        }]
    }
    return settings;
}

@WebSocketGateway({
    cors: {
        origin: '*',
    },
})
export class SignalingGateway implements OnGatewayInit {
    constructor(private readonly mediasoupService: MediasoupService) { }

    afterInit(server: Server) {
        io = server;
        this.mediasoupService.initMediasoup();
        console.log('🚀 Socket.IO Gateway ready');
    }

    handleConnection(socket: Socket) {
        console.log(`User connected ====== : ${socket.id}`);
    }

    handleDisconnect(socket: Socket) {
        console.log(`User disconnected: ${socket.id}`);
        this.closeTransports(socket.id);
        this.closeProducers(socket.id);
        this.closeConsumers(socket.id);
        this.closeTranslationTransports(socket.id);
        this.closeTranslationProducers(socket.id);
        this.closeTranslationConsumers(socket.id);
    }

    @SubscribeMessage('get-rtp-capabilities')
    handleGetRtp(socket: Socket) {
        console.log('Getting RTP capabilities');
        
        const rtpCapabilities = this.mediasoupService.getRtpCapabilities();
        socket.emit('rtp-capabilities', rtpCapabilities);
    }
    
    async setupBidirectionalConnection(socket:Socket, producer:Producer<AppData>, kind:"audio"|"video", targetLang:string): Promise<Producer<AppData> | null>{
        let processedProducer: Producer<AppData> | null = null;
        const sendTransport = await this.mediasoupService.createPlainTransport("send");
        const recvTransport = await this.mediasoupService.createPlainTransport("recv");
        const sessionId = `${socket.id}@${targetLang}`

        const rtpPort = getPort();

        // [Mediasoup -> Track Processor]
        translationTransports.set(`${socket.id}-${kind}-send`, sendTransport);
        await sendTransport.connect({
            ip: '0.0.0.0',
            port: rtpPort,
        });

        const consumer = await sendTransport.consume({
            producerId: producer.id,
            rtpCapabilities: this.mediasoupService.getRtpCapabilities(),
        });
        translationConsumers.set(`${socket.id}-${kind}`, consumer);

        // [Track Processor -> Mediasoup]
        translationTransports.set(`${socket.id}-${kind}-recv`, recvTransport);
        await recvTransport.connect({
            ip: '0.0.0.0',                    // Processing program sends the track to this IP
            port: recvTransport.tuple.localPort,// Processing program sends the track to this port
        });

        const codec = consumer.rtpParameters.codecs[0];
        const payloadType = codec.payloadType;
        const codecName = codec.mimeType.split('/')[1];
        const clockRate = codec.clockRate;
        const channels = codec.channels || 2;
        const ssrc = randomInt(1, 0x7FFFFFFF);
        const connectionSettings = getSettingsForKind(kind, consumer.rtpParameters.codecs);

        let payload = {
            producerId: producer.id,
            rtpPort: rtpPort,
            ip: recvTransport.tuple.localIp,
            codec: codecName,
            clockRate,
            channels,
            payloadType,
            ssrc,
            outputPort: recvTransport.tuple.localPort,
            targetLang,
            sessionId,
            enableVoiceClone: connectionSettings.enableVoiceClone, // when this is updated to be selectable by the user, receive it as a parameter instead
        };

        fetch(connectionSettings.processingServerUrl + connectionSettings.processingServerInitiateMethod, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        })
            .then(response => response.json())
            .then(data => {
                console.log(connectionSettings.successMessage, data);
            })
            .catch(error => {
                console.error(connectionSettings.errorMessage, error);
            });

        // Consume the track data from the track processor
        processedProducer = await recvTransport.produce({
            kind: kind,
            rtpParameters: {
                codecs: connectionSettings.RTPCodecParameter,
                encodings: [{ssrc}]
            },
        });
        translationProducers.set(`${socket.id}-${kind}`, processedProducer);

        // This is only for debugging the blank/dropping video problem
        if (kind === "video") {
            console.log("📹 Video Producer Details:");
            console.log(`  - Producer ID: ${processedProducer.id}`);
            console.log(`  - Kind: ${processedProducer.kind}`);
            console.log(`  - Paused: ${processedProducer.paused}`);
            console.log(`  - Closed: ${processedProducer.closed}`);
            console.log("  - RTP Parameters:");
            console.log(JSON.stringify(processedProducer.rtpParameters, null, 2));
            console.log("  - Associated Transports:");
            console.log(`    - Send Transport ID: ${sendTransport.id}`);
            console.log(`    - Receive Transport ID: ${recvTransport.id}`);
            console.log("  - Session Details:");
            console.log(`    - Session ID: ${sessionId}`);
            console.log(`    - Target Language: ${targetLang}`);
            console.log(`    - RTP Port: ${rtpPort}`);
            console.log(`    - Local Port: ${recvTransport.tuple.localPort}`);
        }

        return processedProducer;
    }

    @SubscribeMessage('create-transport')
    async handleCreateTransport(
        socket: Socket,
        data: { direction: 'send' | 'recv' },
    ) {
        const { direction } = data;
        const { transport, params } =
            await this.mediasoupService.createWebRtcTransport();

        userTransports.set(`${socket.id}-${direction}`, transport);
        socket.emit(`transport-created-${direction}`, params);

        if (direction === 'send') {
            socket.on('connect-transport-send', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-send');
            });

            socket.on('produce', async ({ kind, rtpParameters, roomCode, userId }) => {
                if (kind !== 'audio' && kind !== 'video') {
                    socket.emit('produce-error', `Invalid media kind: ${kind}`);
                    return;
                }
                const targetLang = 'eng'; // Default target language

                const producer = await transport.produce({
                    kind: kind as MediaKind,
                    rtpParameters,
                });

                userProducers.set(`${socket.id}-${kind}`, producer);
                let processedProducer = await this.setupBidirectionalConnection(socket, producer, kind, targetLang)

                socket.join(roomCode);
                if (!rooms.has(roomCode)) {
                    rooms.set(roomCode, { producers: new Map() });
                }
                
                if(processedProducer != null){
                    const producerKey = `${socket.id}-${kind}`;

                    // Logging producer stats for debugging
                    setInterval(async () => {
                        try {
                            const stats = await processedProducer.getStats();
                            console.log(`📊 ${producerKey} transport stats:`, stats);
                        } catch (err) {
                            console.log('Stats error:', err);
                        }
                    }, 5000);

                    rooms.get(roomCode)!.producers.set(producerKey, processedProducer);

                    socket.to(roomCode).emit('new-producer', {
                        producerId: processedProducer.id,
                        socketId: socket.id,
                        kind,
                    });

                    // TODO: something is wrong with the processedProducer for video (wigh drop rate 
                    // when a consumer tries to consume from it, while the original producer workds fine).
                    // try to diff them for debugging

                    if(kind == 'video'){
                        setInterval(async () => {
                            console.log('---------------------------------------------------------')
                            console.log('Original Producer:', {
                                id: producer.id,
                                kind: producer.kind,
                                rtpParameters: producer.rtpParameters,
                                appData: producer.appData,
                                paused: producer.paused,
                                score: producer.score,
                                stats: await producer.getStats(),
                            });

                            console.log('Processed Producer:', {
                                id: processedProducer.id,
                                kind: processedProducer.kind,
                                rtpParameters: processedProducer.rtpParameters,
                                appData: processedProducer.appData,
                                paused: processedProducer.paused,
                                score: processedProducer.score,
                                stats: await processedProducer.getStats(),
                            });
                        }, 5000);
                    }

                }
                else{
                    rooms.get(roomCode)!.producers.set(socket.id, producer);
                    socket.to(roomCode).emit('new-producer', {
                        producerId: producer.id,
                        socketId: socket.id,
                        kind,
                    });
                }

                socket.emit('produced', { id: producer.id });
            });
        } else {
            socket.on('connect-transport-recv', async ({ dtlsParameters }) => {
                await transport.connect({ dtlsParameters });
                socket.emit('transport-connected-recv');
            });
        }
    }

    @SubscribeMessage('consume')
    async handleConsume(
        socket: Socket,
        {
            producerId,
            rtpCapabilities,
        }: {
            producerId: string;
            rtpCapabilities: RtpCapabilities;
        },
    ) {
        const router = this.mediasoupService.getRouter();

        if (!router.canConsume({ producerId, rtpCapabilities })) {
            console.error('❌ Cannot consume this stream');
            socket.emit('consume-error', 'Cannot consume this stream');
            return;
        }

        const transport = userTransports.get(`${socket.id}-recv`);
        if (!transport) {
            console.error('❌ No transport found for receiving media');
            socket.emit('consume-error', 'No transport found');
            return;
        }
        console.log(`✅ Transport connected: ${transport.id}`);

        const consumer = await transport.consume({
            producerId,
            rtpCapabilities,
            paused: false,
        });

        console.log(`✅ Consumer created: ${consumer.id}`);
        console.log(`  - Kind: ${consumer.kind}`);
        console.log(`  - RTP Parameters:`, consumer.rtpParameters);

        socket.emit('consumed', {
            id: consumer.id,
            kind: consumer.kind,
            rtpParameters: consumer.rtpParameters,
            producerId,
        });

        // Logs the stats for debugging the video problem. You can see almost all the frames sent by
        // the python server are dropped, even though the stats from the producer itself seem ok
        setInterval(async () => {
            try {
                const stats = await consumer.getStats();
                console.log(`📊 ${consumer.id} consumer stats:`, stats);
                const Transportstats = await transport.getStats();
                console.log(`📊 ${transport.id} consumer transport stats:`, Transportstats);
            } catch (err) {
                console.log('Stats error:', err);
            }
        }, 5000);

        await consumer.resume();
    }


    closeTransports(socketId: string) {
        const sendTransport = userTransports.get(`${socketId}-send`);
        const recvTransport = userTransports.get(`${socketId}-recv`);
        if (sendTransport) {
            sendTransport.close();
            userTransports.delete(`${socketId}-send`);
        }
        if (recvTransport) {
            recvTransport.close();
            userTransports.delete(`${socketId}-recv`);
        }
    }

    closeProducers(socketId: string) {
        const audioProducer = userProducers.get(socketId + '-audio');
        if (audioProducer) {
            audioProducer.close();
            userProducers.delete(socketId + '-audio');
        }
        const videoProducer = userProducers.get(socketId + '-video');
        if (videoProducer) {
            videoProducer.close();
            userProducers.delete(socketId + '-video');
        }
    }

    closeConsumers(socketId: string) {
        const audioConsumer = userConsumers.get(socketId + '-audio');
        if (audioConsumer) {
            audioConsumer.close();
            userConsumers.delete(socketId + '-audio');
        }
        const videoConsumer = userConsumers.get(socketId + '-video');
        if (videoConsumer) {
            videoConsumer.close();
            userConsumers.delete(socketId + '-video');
        }
    }

    closeTranslationTransports(socketId: string) {
        const sendTransport = translationTransports.get(`${socketId}-send`);
        const recvTransport = translationTransports.get(`${socketId}-recv`);
        const videoTransport = translationTransports.get(`${socketId}-video`);

        if (sendTransport) {
            sendTransport.close();
            translationTransports.delete(`${socketId}-send`);
        }
        if (recvTransport) {
            recvTransport.close();
            translationTransports.delete(`${socketId}-recv`);
        }
        if (videoTransport) {
            videoTransport.close();
            translationTransports.delete(`${socketId}-video`);
        }
    }

    closeTranslationProducers(socketId: string) {
        const audioProducer = translationProducers.get(socketId + '-audio');
        if (audioProducer) {
            audioProducer.close();
            translationProducers.delete(socketId + '-audio');
        }
        const videoProducer = translationProducers.get(socketId + '-video');
        if (videoProducer) {
            videoProducer.close();
            translationProducers.delete(socketId + '-video');
        }
    }

    closeTranslationConsumers(socketId: string) {
        const audioConsumer = translationConsumers.get(socketId + '-audio');
        if (audioConsumer) {
            audioConsumer.close();
            translationConsumers.delete(socketId + '-audio');
        }
        const videoConsumer = translationConsumers.get(socketId + '-video');
        if (videoConsumer) {
            videoConsumer.close();
            translationConsumers.delete(socketId + '-video');
        }
    }

}
