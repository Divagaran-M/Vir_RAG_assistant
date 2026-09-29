import React, { useState, useEffect, useRef } from 'react';
import { ChatMessageData, CampusLocation, BackendHealthStatus } from '../types/chat';
import { ChatMessage } from './ChatMessage';
import { ChatInput } from './ChatInput';
import { TypingIndicator } from './TypingIndicator';
import { QuickQuestions } from './QuickQuestions';
import { aiService } from '../services/aiService';
import { speechService } from '../services/speechService';
import { Volume2, VolumeX, RotateCcw, Sparkles } from 'lucide-react';
import { AnimatePresence } from 'framer-motion';


const INITIAL_MESSAGE: ChatMessageData = {
  id: 'welcome-msg',
  sender: 'ai',
  text: `Hello! 👋 I'm your **College AI Assistant**.\n\nI can help you find information about your college, departments, academics, facilities, placements, events, and campus navigation.\n\nWhat would you like to know?`,
  timestamp: 'Just now',
  categoryTag: 'Welcome',
  suggestedFollowups: [
    'What departments are available?',
    'Where is the IT department?',
    'How does campus placement work?'
  ]
};

interface ChatWindowProps {
  externalQuery?: { text: string; id: number } | null;
  onMapTriggered?: () => void;
}

export const ChatWindow: React.FC<ChatWindowProps> = ({ externalQuery }) => {
  const [messages, setMessages] = useState<ChatMessageData[]>([INITIAL_MESSAGE]);
  const [isThinking, setIsThinking] = useState<boolean>(false);
  const [isVoiceEnabled, setIsVoiceEnabled] = useState<boolean>(speechService.getVoiceEnabled());
  const [selectedVoice, setSelectedVoice] = useState<string>(speechService.getSelectedVoice());
  const availableVoices = speechService.getAvailableVoices();
  const [speakingMessageId, setSpeakingMessageId] = useState<string | undefined>(undefined);
  const [selectedOrigin, setSelectedOrigin] = useState<CampusLocation | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const [health, setHealth] = useState<BackendHealthStatus>({ online: false });

  // Sync speech service speaking state and backend health
  useEffect(() => {
    const unsubscribeSpeech = speechService.addSpeakingListener((isSpeaking, msgId) => {
      setSpeakingMessageId(isSpeaking ? msgId : undefined);
    });
    const unsubscribeHealth = aiService.addHealthListener((status) => {
      setHealth(status);
    });
    return () => {
      unsubscribeSpeech();
      unsubscribeHealth();
    };
  }, []);


  // Handle external query injections from FAQ or Explore Cards
  useEffect(() => {
    if (externalQuery && externalQuery.text) {
      handleSendMessage(externalQuery.text);
    }
  }, [externalQuery]);

  // Scroll to bottom whenever messages change or typing changes
  const scrollToBottom = (smooth: boolean = true) => {
    messagesEndRef.current?.scrollIntoView({ behavior: smooth ? 'smooth' : 'auto' });
  };

  useEffect(() => {
    scrollToBottom(true);
  }, [messages, isThinking]);

  // Toggle Global Voice
  const handleToggleVoice = () => {
    const nextState = !isVoiceEnabled;
    setIsVoiceEnabled(nextState);
    speechService.setVoiceEnabled(nextState);
  };

  // Reset Chat
  const handleResetChat = () => {
    speechService.stopSpeaking();
    setSelectedOrigin(null);
    aiService.resetSession();
    setMessages([
      {
        ...INITIAL_MESSAGE,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
      }
    ]);
  };

  // Core Send Message Handler
  const handleSendMessage = async (text: string) => {
    if (!text.trim() || isThinking) return;

    const userMessageId = 'user-' + Date.now();
    const timeNow = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const newUserMsg: ChatMessageData = {
      id: userMessageId,
      sender: 'user',
      text: text,
      timestamp: timeNow
    };

    // Construct history for multi-turn reasoning
    const currentHistory = messages
      .filter((m) => m.id !== 'welcome-msg')
      .map((m) => ({
        role: (m.sender === 'user' ? 'user' : 'assistant') as 'user' | 'assistant',
        content: m.text
      }));

    setMessages((prev) => [...prev, newUserMsg]);
    setIsThinking(true);

    try {
      // Call AI Service with question and conversation history
      const response = await aiService.sendMessage(text, currentHistory);
      const aiMessageId = 'ai-' + Date.now();

      const newAiMsg: ChatMessageData = {
        id: aiMessageId,
        sender: 'ai',
        text: response.text,
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        categoryTag: response.categoryTag,
        isNavigationPrompt: response.isNavigationPrompt,
        navigationStep: response.navigationStep,
        routeData: response.routeData,
        suggestedFollowups: response.suggestedFollowups,
        debug: response.debug,
        source: response.source,
        citations: response.citations
      };

      setMessages((prev) => [...prev, newAiMsg]);
      setIsThinking(false);

      // Auto Voice Narration if voice enabled
      if (speechService.getVoiceEnabled()) {
        speechService.speak(response.text, aiMessageId);
      }
    } catch (err) {
      console.error('Error getting AI response:', err);
      setIsThinking(false);
    }
  };


  // Navigation Step 1: User Selected Origin
  const handleSelectOrigin = (origin: CampusLocation) => {
    setSelectedOrigin(origin);
    const userMsg: ChatMessageData = {
      id: 'user-' + Date.now(),
      sender: 'user',
      text: `I am currently at ${origin.name}.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const destPrompt = aiService.getDestinationPrompt(origin.name);
    const aiMsg: ChatMessageData = {
      id: 'ai-' + Date.now(),
      sender: 'ai',
      text: destPrompt.text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isNavigationPrompt: true,
      navigationStep: 'select_destination',
      categoryTag: destPrompt.categoryTag
    };

    setMessages((prev) => [...prev, userMsg, aiMsg]);

    if (speechService.getVoiceEnabled()) {
      speechService.speak(`Great! You are at ${origin.name}. Where do you want to go, or whom do you want to see?`, aiMsg.id);
    }
  };

  // Navigation Step 2: User Selected Destination
  const handleSelectDestination = (destination: CampusLocation) => {
    const origin = selectedOrigin || { id: 'main-gate', name: 'Main Gate' } as any;

    const userMsg: ChatMessageData = {
      id: 'user-' + Date.now(),
      sender: 'user',
      text: `I want to go to ${destination.name}.`,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    };

    const routeResult = aiService.getRouteResult(origin.id, destination.id);
    const aiMsg: ChatMessageData = {
      id: 'ai-' + Date.now(),
      sender: 'ai',
      text: routeResult.text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      isNavigationPrompt: true,
      navigationStep: 'route_found',
      routeData: routeResult.routeData,
      categoryTag: routeResult.categoryTag,
      suggestedFollowups: routeResult.suggestedFollowups
    };

    setMessages((prev) => [...prev, userMsg, aiMsg]);

    if (speechService.getVoiceEnabled() && routeResult.routeData) {
      const speechText = `The ${destination.name} is approximately ${routeResult.routeData.walkingTimeMinutes} minutes away from the ${origin.name}.`;
      speechService.speak(speechText, aiMsg.id);
    }
  };

  // Quick Question Selection
  const handleSelectQuickQuestion = (query: string, isMapAction?: boolean) => {
    if (isMapAction) {
      handleSendMessage('Campus Map');
    } else {
      handleSendMessage(query);
    }
  };

  return (
    <div
      id="chatbot"
      className="bg-white rounded-3xl border border-slate-200 shadow-card hover:shadow-elevated transition-shadow duration-300 flex flex-col h-[750px] sm:h-[800px] overflow-hidden"
    >
      {/* Chatbot Header */}
      <div className="bg-white border-b border-slate-200 px-4 sm:px-6 py-4 flex items-center justify-between flex-shrink-0 z-10">
        {/* Left: Assistant Identity & Live Status */}
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-10 h-10 rounded-2xl bg-blue-600 text-white flex items-center justify-center shadow-md shadow-blue-600/20">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            {/* Online / Standby Dot */}
            <span
              className={`absolute -bottom-0.5 -right-0.5 w-3.5 h-3.5 rounded-full ring-2 ring-white ${
                health.online ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
              }`}
            />
          </div>

          <div>
            <div className="flex items-center gap-2">
              <h3 className="font-extrabold text-slate-900 text-base leading-tight">
                College AI Assistant
              </h3>
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full uppercase tracking-wider flex items-center gap-1 border ${
                  health.online
                    ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                    : 'bg-amber-50 text-amber-700 border-amber-200'
                }`}
                title={health.online ? `Connected: ${health.service}` : 'Standby / Local Fallback'}
              >
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    health.online ? 'bg-emerald-500' : 'bg-amber-500'
                  }`}
                />
                {health.online ? 'ONLINE' : 'STANDBY'}
              </span>
            </div>
            <p className="text-xs text-slate-500 font-medium">
              {health.online
                ? 'RAG Agent Active • P.T. Lee CNCET'
                : 'Always here to help • P.T. Lee CNCET'}
            </p>
          </div>
        </div>


        {/* Right: Controls (Voice Toggle & Reset) */}
        <div className="flex items-center gap-2">
          {/* Global Voice Toggle Button */}
          <button
            onClick={handleToggleVoice}
            aria-label={isVoiceEnabled ? "Mute automatic AI voice readout" : "Enable automatic AI voice readout"}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-bold transition-all shadow-2xs cursor-pointer ${
              isVoiceEnabled
                ? 'bg-blue-50 text-blue-700 border border-blue-200 hover:bg-blue-100'
                : 'bg-slate-100 text-slate-600 border border-slate-200 hover:bg-slate-200'
            }`}
            title={isVoiceEnabled ? 'Voice output is ON. Click to mute' : 'Voice output is OFF. Click to unmute'}
          >
            {isVoiceEnabled ? (
              <>
                <Volume2 className="w-3.5 h-3.5 text-blue-600 animate-pulse" />
                <span className="hidden sm:inline">Voice ON</span>
              </>
            ) : (
              <>
                <VolumeX className="w-3.5 h-3.5 text-slate-400" />
                <span className="hidden sm:inline">Voice OFF</span>
              </>
            )}
          </button>

          {/* Voice Model Selector (when voice enabled) */}
          {isVoiceEnabled && (
            <select
              value={selectedVoice}
              onChange={(e) => {
                const newVoice = e.target.value;
                setSelectedVoice(newVoice);
                speechService.setSelectedVoice(newVoice);
              }}
              title="Select AI Neural Voice"
              aria-label="Select AI Neural Voice"
              className="bg-white/95 border border-blue-200 text-blue-900 text-xs font-semibold rounded-xl px-2.5 py-1.5 hover:bg-blue-50/50 transition-colors shadow-2xs focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer max-w-[130px] sm:max-w-[170px] truncate"
            >
              {availableVoices.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name} ({v.accent})
                </option>
              ))}
            </select>
          )}

          {/* Reset Conversation Button */}
          <button
            onClick={handleResetChat}
            aria-label="Restart conversation"
            className="p-2 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
            title="Reset conversation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Messages Scroll Area */}
      <div
        ref={chatContainerRef}
        className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-2 bg-[#F8FAFC]/60 scroll-smooth"
      >
        {messages.map((message) => (
          <ChatMessage
            key={message.id}
            message={message}
            isSpeaking={speakingMessageId === message.id}
            onSelectOrigin={handleSelectOrigin}
            onSelectDestination={handleSelectDestination}
            onSuggestedPrompt={handleSendMessage}
            selectedOriginName={selectedOrigin?.name}
          />
        ))}

        {/* Thinking Indicator */}
        <AnimatePresence>
          {isThinking && <TypingIndicator />}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Footer Area: Input & Quick Questions */}
      <div className="p-4 sm:p-5 bg-white border-t border-slate-200 space-y-3 flex-shrink-0">
        <ChatInput
          onSendMessage={handleSendMessage}
          disabled={isThinking}
        />

        <QuickQuestions
          onSelectQuestion={handleSelectQuickQuestion}
          disabled={isThinking}
        />
      </div>
    </div>
  );
};
