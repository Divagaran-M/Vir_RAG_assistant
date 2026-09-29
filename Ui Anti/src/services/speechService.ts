// Speech Service handling Microphone SpeechRecognition and Microsoft Edge Neural Text-to-Speech

export interface SpeechRecognitionResultCallback {
  (text: string, isFinal: boolean): void;
}

export interface SpeechRecognitionErrorCallback {
  (error: string): void;
}

export interface EdgeVoiceOption {
  id: string;
  name: string;
  gender: string;
  locale: string;
  accent: string;
  recommended?: boolean;
}

export const POPULAR_EDGE_VOICES: EdgeVoiceOption[] = [
  {
    id: 'en-IN-NeerjaNeural',
    name: 'Neerja',
    gender: 'Female',
    locale: 'en-IN',
    accent: 'Indian English',
    recommended: true
  },
  {
    id: 'en-IN-PrabhatNeural',
    name: 'Prabhat',
    gender: 'Male',
    locale: 'en-IN',
    accent: 'Indian English',
    recommended: true
  },
  {
    id: 'en-US-JennyNeural',
    name: 'Jenny',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US English',
    recommended: false
  },
  {
    id: 'en-US-GuyNeural',
    name: 'Guy',
    gender: 'Male',
    locale: 'en-US',
    accent: 'US English',
    recommended: false
  },
  {
    id: 'en-US-AriaNeural',
    name: 'Aria',
    gender: 'Female',
    locale: 'en-US',
    accent: 'US English',
    recommended: false
  },
  {
    id: 'en-GB-SoniaNeural',
    name: 'Sonia',
    gender: 'Female',
    locale: 'en-GB',
    accent: 'UK English',
    recommended: false
  },
  {
    id: 'ta-IN-PallaviNeural',
    name: 'Pallavi',
    gender: 'Female',
    locale: 'ta-IN',
    accent: 'Tamil',
    recommended: false
  }
];

class SpeechService {
  private isVoiceEnabled: boolean = true;
  private selectedVoice: string = 'en-IN-NeerjaNeural';
  private currentUtterance: SpeechSynthesisUtterance | null = null;
  private currentAudio: HTMLAudioElement | null = null;
  private currentAudioUrl: string | null = null;
  private abortController: AbortController | null = null;
  private recognition: any = null;
  private isRecognizing: boolean = false;
  private onSpeakingStateChangeListeners: ((speaking: boolean, currentMessageId?: string) => void)[] = [];
  private activeSpeakingMessageId?: string;

  constructor() {
    try {
      const savedEnabled = localStorage.getItem('college_ai_voice_enabled');
      this.isVoiceEnabled = savedEnabled !== null ? savedEnabled === 'true' : true;

      const savedVoice = localStorage.getItem('college_ai_edge_voice');
      if (savedVoice) {
        this.selectedVoice = savedVoice;
      }
    } catch {
      this.isVoiceEnabled = true;
    }

    this.initRecognition();
  }

  public getVoiceEnabled(): boolean {
    return this.isVoiceEnabled;
  }

  public setVoiceEnabled(enabled: boolean): void {
    this.isVoiceEnabled = enabled;
    try {
      localStorage.setItem('college_ai_voice_enabled', String(enabled));
    } catch {
      // ignore
    }
    if (!enabled) {
      this.stopSpeaking();
    }
  }

  public getSelectedVoice(): string {
    return this.selectedVoice;
  }

  public setSelectedVoice(voiceId: string): void {
    this.selectedVoice = voiceId;
    try {
      localStorage.setItem('college_ai_edge_voice', voiceId);
    } catch {
      // ignore
    }
    // Stop current speech if active so next speak uses new voice
    if (this.isCurrentlySpeaking()) {
      this.stopSpeaking();
    }
  }

  public getAvailableVoices(): EdgeVoiceOption[] {
    return POPULAR_EDGE_VOICES;
  }

  private initRecognition() {
    if (typeof window === 'undefined') return;

    const SpeechRecognition =
      (window as any).SpeechRecognition ||
      (window as any).webkitSpeechRecognition ||
      (window as any).mozSpeechRecognition ||
      (window as any).msSpeechRecognition;

    if (SpeechRecognition) {
      try {
        this.recognition = new SpeechRecognition();
        this.recognition.continuous = false;
        this.recognition.interimResults = true;
        this.recognition.lang = 'en-US';
      } catch (err) {
        console.warn('SpeechRecognition initialization error:', err);
      }
    }
  }

  public isSpeechRecognitionSupported(): boolean {
    return typeof window !== 'undefined' && (
      'SpeechRecognition' in window ||
      'webkitSpeechRecognition' in window ||
      'mozSpeechRecognition' in window ||
      'msSpeechRecognition' in window
    );
  }

  public startListening(
    onResult: SpeechRecognitionResultCallback,
    onError?: SpeechRecognitionErrorCallback,
    onEnd?: () => void
  ): boolean {
    if (!this.isSpeechRecognitionSupported() || !this.recognition) {
      if (onError) onError('Voice input is not supported in this browser. You can type your question instead.');
      return false;
    }

    if (this.isRecognizing) {
      this.stopListening();
    }

    this.stopSpeaking();

    this.recognition.onstart = () => {
      this.isRecognizing = true;
    };

    this.recognition.onresult = (event: any) => {
      let interimTranscript = '';
      let finalTranscript = '';

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interimTranscript += event.results[i][0].transcript;
        }
      }

      const text = finalTranscript || interimTranscript;
      onResult(text, Boolean(finalTranscript));
    };

    this.recognition.onerror = (event: any) => {
      this.isRecognizing = false;
      let msg = 'Voice recognition error occurred.';
      if (event.error === 'not-allowed') {
        msg = 'Microphone permission was denied. Please allow microphone access in your browser.';
      } else if (event.error === 'no-speech') {
        msg = 'No speech was detected. Please try speaking again.';
      }
      if (onError) onError(msg);
    };

    this.recognition.onend = () => {
      this.isRecognizing = false;
      if (onEnd) onEnd();
    };

    try {
      this.recognition.start();
      return true;
    } catch (err) {
      console.warn('Error starting speech recognition:', err);
      this.isRecognizing = false;
      if (onError) onError('Unable to start microphone.');
      return false;
    }
  }

  public stopListening(): void {
    if (this.recognition && this.isRecognizing) {
      try {
        this.recognition.stop();
      } catch {
        // ignore
      }
      this.isRecognizing = false;
    }
  }

  public isSpeechSynthesisSupported(): boolean {
    return typeof window !== 'undefined' && ('Audio' in window || 'speechSynthesis' in window);
  }

  public addSpeakingListener(listener: (speaking: boolean, currentMessageId?: string) => void): () => void {
    this.onSpeakingStateChangeListeners.push(listener);
    return () => {
      this.onSpeakingStateChangeListeners = this.onSpeakingStateChangeListeners.filter(l => l !== listener);
    };
  }

  private notifySpeakingState(speaking: boolean, messageId?: string) {
    this.activeSpeakingMessageId = speaking ? messageId : undefined;
    this.onSpeakingStateChangeListeners.forEach(listener => listener(speaking, messageId));
  }

  public cleanTextForSpeech(rawText: string): string {
    return rawText
      // Remove code blocks
      .replace(/```[\s\S]*?```/g, ' [code block] ')
      // Convert markdown links [text](url) to text
      .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      // Remove emojis
      .replace(/[\u{1F600}-\u{1F6FF}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}]/gu, '')
      // Remove markdown formatting symbols
      .replace(/[*_~`#>\-•]/g, ' ')
      // Expand common college abbreviations
      .replace(/\(B\.E\. \/ B\.Tech\)/gi, 'B E and B Tech')
      .replace(/P\.T\. Lee CNCET/gi, 'P T Lee College of Engineering')
      .replace(/\bCSE\b/gi, 'Computer Science and Engineering')
      .replace(/\bIT\b/gi, 'Information Technology')
      .replace(/\bECE\b/gi, 'Electronics and Communication')
      .replace(/\bEEE\b/gi, 'Electrical and Electronics')
      .replace(/\bMECH\b/gi, 'Mechanical Engineering')
      .replace(/\s+/g, ' ')
      .trim();
  }

  private cleanupCurrentAudio(): void {
    if (this.currentAudio) {
      try {
        this.currentAudio.pause();
        this.currentAudio.currentTime = 0;
        this.currentAudio.onplay = null;
        this.currentAudio.onended = null;
        this.currentAudio.onerror = null;
      } catch {
        // ignore
      }
      this.currentAudio = null;
    }
    if (this.currentAudioUrl) {
      try {
        URL.revokeObjectURL(this.currentAudioUrl);
      } catch {
        // ignore
      }
      this.currentAudioUrl = null;
    }
  }

  /**
   * Speak given text using Microsoft Edge Neural TTS with browser WebSpeech fallback.
   */
  public async speak(text: string, messageId?: string, force: boolean = false): Promise<void> {
    if (!this.isVoiceEnabled && !force) return;

    this.stopSpeaking();

    const cleanText = this.cleanTextForSpeech(text);
    if (!cleanText) return;

    // Set active message id & notify speaking animation
    this.notifySpeakingState(true, messageId);

    const controller = new AbortController();
    this.abortController = controller;

    try {
      const response = await fetch('/api/tts', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          text: cleanText,
          voice: this.selectedVoice,
          rate: '+0%',
          pitch: '+0Hz'
        }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`Edge TTS API returned HTTP status ${response.status}`);
      }

      const blob = await response.blob();
      if (controller.signal.aborted) {
        return;
      }

      this.cleanupCurrentAudio();

      const audioUrl = URL.createObjectURL(blob);
      this.currentAudioUrl = audioUrl;

      const audio = new Audio(audioUrl);
      this.currentAudio = audio;

      audio.onended = () => {
        this.cleanupCurrentAudio();
        this.notifySpeakingState(false);
      };

      audio.onerror = (e) => {
        console.warn('Edge TTS audio playback error:', e);
        this.cleanupCurrentAudio();
        this.notifySpeakingState(false);
      };

      await audio.play();
    } catch (err: any) {
      if (err.name === 'AbortError') {
        // Playback was cancelled by the user
        return;
      }
      console.warn('Microsoft Edge TTS failed, falling back to WebSpeech API:', err);
      this.speakFallbackBrowser(cleanText, messageId);
    }
  }

  /**
   * Fallback to browser SpeechSynthesis if the Edge TTS backend service is unreachable.
   */
  private speakFallbackBrowser(cleanText: string, messageId?: string): void {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      this.notifySpeakingState(false);
      return;
    }

    try {
      const utterance = new SpeechSynthesisUtterance(cleanText);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      utterance.lang = 'en-US';

      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(
        v => v.lang.startsWith('en') && (v.name.includes('Natural') || v.name.includes('Google') || v.name.includes('Samantha') || v.name.includes('Zira'))
      ) || voices.find(v => v.lang.startsWith('en'));

      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      utterance.onstart = () => {
        this.notifySpeakingState(true, messageId);
      };

      utterance.onend = () => {
        this.notifySpeakingState(false);
        this.currentUtterance = null;
      };

      utterance.onerror = (e) => {
        if (e.error !== 'interrupted' && e.error !== 'canceled') {
          console.warn('Speech synthesis error:', e);
        }
        this.notifySpeakingState(false);
        this.currentUtterance = null;
      };

      this.currentUtterance = utterance;
      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn('Speech synthesis invocation failed:', err);
      this.notifySpeakingState(false);
    }
  }

  public stopSpeaking(): void {
    if (this.abortController) {
      try {
        this.abortController.abort();
      } catch {
        // ignore
      }
      this.abortController = null;
    }

    this.cleanupCurrentAudio();

    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel();
      } catch {
        // ignore
      }
      this.currentUtterance = null;
    }

    this.notifySpeakingState(false);
  }

  public isCurrentlySpeaking(): boolean {
    const isAudioActive = Boolean(this.currentAudio && !this.currentAudio.paused && !this.currentAudio.ended);
    const isUtteranceActive = Boolean(typeof window !== 'undefined' && 'speechSynthesis' in window && window.speechSynthesis.speaking && !window.speechSynthesis.paused);
    const isFetching = Boolean(this.abortController);

    return isAudioActive || isUtteranceActive || isFetching;
  }

  public getActiveSpeakingMessageId(): string | undefined {
    return this.activeSpeakingMessageId;
  }
}

export const speechService = new SpeechService();
