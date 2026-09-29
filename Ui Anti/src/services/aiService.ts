import { NavigationRoute, ChatDebugInfo, BackendHealthStatus } from '../types/chat';
import { MOCK_RESPONSES, DEFAULT_FALLBACK_RESPONSE } from '../data/mockResponses';
import { FAQ_DATA } from '../data/faq';
import { CAMPUS_DESTINATIONS, CAMPUS_ORIGINS, calculateRoute } from '../data/campusLocations';

export interface AIResponsePayload {
  text: string;
  categoryTag?: string;
  isNavigationPrompt?: boolean;
  navigationStep?: 'idle' | 'select_current' | 'select_destination' | 'route_found';
  routeData?: NavigationRoute;
  suggestedFollowups?: string[];
  debug?: ChatDebugInfo;
  source?: string;
  citations?: string[];
  isBackendConnected?: boolean;
}

export interface ChatHistoryItem {
  role: 'user' | 'assistant';
  content: string;
}

type HealthListener = (status: BackendHealthStatus) => void;

class AIService {
  private sessionId: string;
  private backendBaseUrl: string;
  private healthListeners: Set<HealthListener> = new Set();
  private lastHealthStatus: BackendHealthStatus = {
    online: true,
    status: 'healthy',
    service: 'Vir Campus Assistant RAG API',
    model: 'Groq Llama 3 / Agent',
    version: '2.0.0'
  };

  constructor() {
    this.sessionId = this.initSessionId();
    // Use relative path by default to leverage Vite proxy, with direct fallback option
    this.backendBaseUrl = (import.meta as any).env?.VITE_API_URL || '';
  }

  /**
   * Initializes or loads a persistent UUID session ID
   */
  private initSessionId(): string {
    const storageKey = 'vir_ai_session_id';
    try {
      const existing = localStorage.getItem(storageKey);
      if (existing) return existing;
    } catch {
      // localStorage may fail in private mode
    }

    const newId = 'vir-sess-' + crypto.randomUUID();
    try {
      localStorage.setItem(storageKey, newId);
    } catch {
      // ignore
    }
    return newId;
  }

  /**
   * Returns the current session ID
   */
  public getSessionId(): string {
    return this.sessionId;
  }

  /**
   * Resets the conversation session ID
   */
  public resetSession(): string {
    const newId = 'vir-sess-' + crypto.randomUUID();
    this.sessionId = newId;
    try {
      localStorage.setItem('vir_ai_session_id', newId);
    } catch {
      // ignore
    }
    return this.sessionId;
  }

  /**
   * Subscribe to backend health status updates
   */
  public addHealthListener(listener: HealthListener): () => void {
    this.healthListeners.add(listener);
    listener(this.lastHealthStatus);
    return () => {
      this.healthListeners.delete(listener);
    };
  }

  /**
   * Returns current backend connection status without calling /health endpoints.
   */
  public async checkBackendHealth(): Promise<BackendHealthStatus> {
    return this.lastHealthStatus;
  }

  private notifyHealthListeners(status: BackendHealthStatus) {
    this.healthListeners.forEach(listener => {
      try {
        listener(status);
      } catch (err) {
        console.error('Error in health listener:', err);
      }
    });
  }

  /**
   * Main entry point for dispatching a user message to the backend RAG engine.
   */
  public async sendMessage(
    message: string,
    history: ChatHistoryItem[] = [],
    _currentNavState?: 'idle' | 'select_current' | 'select_destination' | 'route_found'
  ): Promise<AIResponsePayload> {
    const normalized = message.toLowerCase().trim();

    // 1. Direct interactive campus navigation triggers
    if (
      normalized === 'campus map' ||
      normalized === 'map' ||
      normalized.includes('campus navigation') ||
      normalized.includes('show map') ||
      normalized.includes('show campus map')
    ) {
      return {
        text: `Sure! I can help you navigate the campus. 📍\n\nWhere are you now?`,
        isNavigationPrompt: true,
        navigationStep: 'select_current',
        categoryTag: 'Campus Navigation',
        suggestedFollowups: ['From Main Gate', 'From Admin Block', 'From Canteen']
      };
    }

    // Check for direct route keywords e.g. "where is the IT department from main gate"
    const directDest = CAMPUS_DESTINATIONS.find(d => 
      normalized.includes(d.name.toLowerCase()) || 
      normalized.includes(d.id.replace('-', ' '))
    );

    if (directDest && normalized.includes('from main gate')) {
      const origin = CAMPUS_ORIGINS.find(o => o.id === 'main-gate')!;
      const route = calculateRoute(origin, directDest);
      return {
        text: `📍 **Route Found to ${directDest.name}**\n\nStarting from **${origin.name}**, the **${directDest.name}** is approximately **${route.distanceMeters} meters** away (${route.walkingTimeMinutes} min walk).\n\n${directDest.description}`,
        isNavigationPrompt: true,
        navigationStep: 'route_found',
        routeData: route,
        categoryTag: 'Campus Navigation',
        suggestedFollowups: ['Check another location', 'What courses are offered?', 'Who is the HOD?']
      };
    }

    // 2. Attempt Real Backend RAG API Call
    try {
      const backendResponse = await this.callBackendChat(message, history);
      if (backendResponse) {
        return backendResponse;
      }
    } catch (backendError) {
      console.warn('[AIService] Backend API call failed, using graceful local fallback:', backendError);
    }

    // 3. Fallback to Local Knowledge Base / Mock Templates
    return this.getLocalFallbackResponse(message);
  }

  /**
   * Calls the FastAPI backend /chat or /api/chat endpoint
   */
  private async callBackendChat(
    question: string,
    history: ChatHistoryItem[]
  ): Promise<AIResponsePayload | null> {
    const candidateEndpoints = this.backendBaseUrl
      ? [`${this.backendBaseUrl}/chat`, `${this.backendBaseUrl}/api/chat`]
      : ['/chat', '/api/chat', 'http://127.0.0.1:8000/chat', 'http://127.0.0.1:8000/api/chat'];

    const payload = {
      question: question.trim(),
      filename: '',
      history: history.slice(-6).map(h => ({
        role: h.role,
        content: h.content
      })),
      session_id: this.sessionId
    };

    for (const endpoint of candidateEndpoints) {
      try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 60000); // 60s timeout for multi-agent loops

        const res = await fetch(endpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(payload),
          signal: controller.signal
        });
        clearTimeout(timeoutId);

        if (res.ok) {
          const data = await res.json();
          const answerText = data.answer || data.message || '';

          // Parse citations from text or payload
          const citations: string[] = [];
          const sourceMatch = answerText.match(/\*\*Sources:\*\*\s*(.+)/i);
          if (sourceMatch && sourceMatch[1]) {
            citations.push(sourceMatch[1].trim());
          }

          // Determine category tag from tools or source
          let categoryTag = 'Campus Assistant';
          const toolsUsed: string[] = data.debug?.tools_used || [];
          if (toolsUsed.includes('sql_query') || data.source === 'fast_path_sql') {
            categoryTag = 'Student & Academics DB';
          } else if (toolsUsed.includes('vector_search')) {
            categoryTag = 'Regulations & Docs';
          } else if (toolsUsed.includes('find_path') || toolsUsed.includes('list_rooms') || data.source === 'fast_path_map') {
            categoryTag = 'Campus Navigation';
          } else if (data.source === 'agent') {
            categoryTag = 'AI Agentic RAG';
          }

          // Check if response contains navigation directions and matches a campus destination
          let routeData: NavigationRoute | undefined = undefined;
          let isNavigationPrompt = false;
          let navigationStep: 'idle' | 'select_current' | 'select_destination' | 'route_found' | undefined = undefined;

          if (toolsUsed.includes('find_path') || categoryTag === 'Campus Navigation') {
            const matchedDest = CAMPUS_DESTINATIONS.find(d => 
              answerText.toLowerCase().includes(d.name.toLowerCase()) || 
              question.toLowerCase().includes(d.name.toLowerCase())
            );
            if (matchedDest) {
              const defaultOrigin = CAMPUS_ORIGINS.find(o => o.id === 'main-gate') || CAMPUS_ORIGINS[0];
              routeData = calculateRoute(defaultOrigin, matchedDest);
              isNavigationPrompt = true;
              navigationStep = 'route_found';
            }
          }

          this.lastHealthStatus = {
            online: true,
            status: 'healthy',
            service: 'Vir Campus Assistant RAG API',
            model: 'Groq Llama 3 / Agent',
            version: '2.0.0'
          };
          this.notifyHealthListeners(this.lastHealthStatus);

          return {
            text: answerText,
            categoryTag,
            suggestedFollowups: data.followups && data.followups.length > 0 ? data.followups : undefined,
            debug: data.debug,
            source: data.source,
            citations: citations.length > 0 ? citations : undefined,
            isBackendConnected: true,
            isNavigationPrompt,
            navigationStep,
            routeData
          };
        }
      } catch (err) {
        // try next endpoint candidate
        continue;
      }
    }

    this.lastHealthStatus = {
      online: false,
      status: 'offline',
      message: 'Backend disconnected (using local fallback)'
    };
    this.notifyHealthListeners(this.lastHealthStatus);

    return null;
  }

  /**
   * Local fallback engine when backend is offline
   */
  private async getLocalFallbackResponse(message: string): Promise<AIResponsePayload> {
    const normalized = message.toLowerCase().trim();

    // 1. Search against FAQ Knowledge Base
    const matchedFaq = FAQ_DATA.find(faq => {
      const q = faq.question.toLowerCase();
      const tokens = normalized.split(/\s+/).filter(t => t.length > 2);
      const matchesTokens = tokens.some(t => q.includes(t) || faq.tags.some(tag => tag.toLowerCase().includes(t)));
      return q.includes(normalized) || matchesTokens;
    });

    if (matchedFaq && normalized.length > 6) {
      return {
        text: matchedFaq.answer,
        categoryTag: matchedFaq.category,
        suggestedFollowups: matchedFaq.tags.slice(0, 3).map(t => `Tell me more about ${t}`),
        isBackendConnected: false
      };
    }

    // 2. Search Mock Templates with weighted keyword score
    let bestTemplate = MOCK_RESPONSES[0];
    let maxMatches = 0;

    for (const template of MOCK_RESPONSES) {
      let matches = 0;
      for (const keyword of template.keywords) {
        if (normalized.includes(keyword.toLowerCase())) {
          matches += keyword.length > 4 ? 2 : 1;
        }
      }
      if (matches > maxMatches) {
        maxMatches = matches;
        bestTemplate = template;
      }
    }

    if (maxMatches > 0) {
      if (bestTemplate.isNavigationIntent) {
        return {
          text: bestTemplate.reply,
          isNavigationPrompt: true,
          navigationStep: 'select_current',
          categoryTag: 'Campus Navigation',
          isBackendConnected: false
        };
      }

      return {
        text: bestTemplate.reply,
        categoryTag: bestTemplate.categoryTag,
        suggestedFollowups: bestTemplate.suggestedFollowups,
        isBackendConnected: false
      };
    }

    // 3. General Fallback
    return {
      text: DEFAULT_FALLBACK_RESPONSE,
      categoryTag: 'General Assistant',
      suggestedFollowups: [
        'What departments are available?',
        'How does placement work?',
        'Campus Map'
      ],
      isBackendConnected: false
    };
  }

  public getDestinationPrompt(originName: string): AIResponsePayload {
    return {
      text: `Great! You are at **${originName}**.\n\nWhere do you want to go, or whom do you want to see?`,
      isNavigationPrompt: true,
      navigationStep: 'select_destination',
      categoryTag: 'Campus Navigation'
    };
  }

  public getRouteResult(fromId: string, toId: string): AIResponsePayload {
    const fromLoc = CAMPUS_ORIGINS.find(o => o.id === fromId) || CAMPUS_ORIGINS[0];
    const toLoc = CAMPUS_DESTINATIONS.find(d => d.id === toId) || CAMPUS_DESTINATIONS[0];

    const route = calculateRoute(fromLoc, toLoc);

    return {
      text: `📍 **Route Found!**\n\nThe **${toLoc.name}** is approximately **${route.walkingTimeMinutes} minutes** (${route.distanceMeters} meters) away from **${fromLoc.name}**.\n\n${toLoc.floor ? `📌 *Location Details*: ${toLoc.floor}\n` : ''}${toLoc.description}`,
      isNavigationPrompt: true,
      navigationStep: 'route_found',
      routeData: route,
      categoryTag: 'Campus Navigation',
      suggestedFollowups: [
        'How do I find another department?',
        'Tell me about the facilities at ' + toLoc.name,
        'Where is the canteen?'
      ]
    };
  }
}

export const aiService = new AIService();
