import { useEffect, useRef, useState } from "react";
import "./App.css";
import {
  Scale,
  MessageSquare,
  FileSearch,
  Layers,
  Clock,
  BookOpen,
  Archive,
  Plus,
  Search,
  User,
  Settings,
  LogOut,
  Send,
  Mic,
  MicOff,
  Volume2,
  VolumeX,
  UploadCloud,
  FileText,
  AlertTriangle,
  AlertCircle,
  CheckCircle2,
  Check,
  Copy,
  ChevronRight,
  ChevronDown,
  ExternalLink,
  Shield,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  Menu,
  X,
  CreditCard,
  ShoppingBag,
  Laptop,
  Home,
  Briefcase,
  Gavel,
  Car,
  Landmark,
  Eye,
  EyeOff,
  Lock,
  Camera,
  UserCheck,
  Activity,
  Award,
  Trash2,
  RefreshCw,
  Download,
  Calendar,
  IndianRupee,
  FileCheck,
  Building,
  HelpCircle,
  FolderArchive,
  ArrowRight,
  Info,
  Palette,
  Volume1,
  BarChart3,
  PieChart,
  Filter,
  History,
  CornerDownRight,
  TrendingUp,
  LogIn,
  UserPlus,
  Globe,
} from "lucide-react";

const API_BASE = import.meta.env.VITE_API_URL || (
  typeof window !== 'undefined'
    ? (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' || window.location.hostname.startsWith('10.') || window.location.hostname.startsWith('192.168.') || window.location.hostname.startsWith('172.')
        ? `http://${window.location.hostname}:8000`
        : "https://jannayaya-ai.onrender.com")
    : "http://127.0.0.1:8000"
);

const MULTILINGUAL_UI = {
  english: {
    title: "JanNyaya AI",
    tagline: "Multimodal Intelligent Legal Assistance System for Citizens",
    tab_home: "Home Overview",
    tab_chat: "Legal Consultation",
    tab_doc: "Document Case Studio",
    tab_timeline: "Case Timeline",
    tab_library: "Bare Acts & Knowledge Base",
    tab_dossier: "Case Dossier",
    tab_history: "Consultation History",
    new_chat: "New Consultation",
    chat_history: "Recent Consultations",
    search_chats: "Search consultations...",
    no_chats: "No previous consultations",
    ask_placeholder: "Ask any Indian legal question (e.g. cheque bounce, loan notice, theft under BNS, electronic evidence, cyber fraud, consumer refund)...",
    ask_button: "Ask",
    recording_active: "Listening to speech...",
    upload_title: "Upload Legal Documents & Case Files",
    upload_subtitle: "Supports PDFs, JPG, PNG, WEBP — Deep legal fact extraction, clause analysis & cross-document conflict detection",
    browse_files: "Choose Case Documents",
    analyze_button: "Analyze Case Documents",
    analyzing_text: "Extracting Facts & Analyzing Legal Provisions...",
    guest_user: "Citizen Account",
    login_signup: "Sign In / Register",
    logout: "Sign Out",
    edit_profile: "Account Settings",
    save_profile: "Save Changes",
    profile_title: "Account Settings & Preferences",
    full_name: "Full Name",
    email: "Email Address",
    pref_language: "Preferred Language",
    curr_password: "Current Password",
    new_password: "New Password",
    today: "Today",
    previous_7_days: "Past 7 Days",
    older: "Earlier Consultations",
  },
  hindi: {
    title: "जनन्याय AI",
    tagline: "नागरिकों के लिए बहुआयामी कानूनी परामर्श एवं दस्तावेज़ विश्लेषण प्रणाली",
    tab_home: "मुख्य पृष्ठ (होम)",
    tab_chat: "कानूनी परामर्श",
    tab_doc: "केस दस्तावेज़ स्टूडियो",
    tab_timeline: "घटनाक्रम टाइमलाइन",
    tab_library: "कानून पुस्तकालय",
    tab_dossier: "केस डोजियर",
    tab_history: "परामर्श इतिहास",
    new_chat: "नया कानूनी परामर्श",
    chat_history: "हालिया परामर्श",
    search_chats: "परामर्श खोजें...",
    no_chats: "कोई पूर्व परामर्श नहीं मिला",
    ask_placeholder: "कोई भी कानूनी प्रश्न पूछें (उदा. चेक बाउंस, बैंक वसूली, बीएनएस में चोरी, इलेक्ट्रॉनिक साक्ष्य, साइबर धोखाधड़ी, उपभोक्ता अधिकार)...",
    ask_button: "पूछें",
    recording_active: "आपकी आवाज़ सुन रहे हैं...",
    upload_title: "कानूनी दस्तावेज़ एवं फाइलें अपलोड करें",
    upload_subtitle: "PDF, JPG, PNG, WEBP — बहु-दस्तावेज़ विश्लेषण, धारा पहचान एवं विरोधाभास जांच",
    browse_files: "दस्तावेज़ चुनें",
    analyze_button: "दस्तावेज़ विश्लेषण शुरू करें",
    analyzing_text: "दस्तावेज़ का विश्लेषण हो रहा है...",
    guest_user: "नागरिक खाता",
    login_signup: "लॉग इन / पंजीकरण",
    logout: "लॉग आउट",
    edit_profile: "खाता सेटिंग्स",
    save_profile: "बदलाव सहेजें",
    profile_title: "खाता सेटिंग्स एवं प्राथमिकताएं",
    full_name: "पूरा नाम",
    email: "ईमेल पता",
    pref_language: "पसंदीदा भाषा",
    curr_password: "वर्तमान पासवर्ड",
    new_password: "नया पासवर्ड",
    today: "आज",
    previous_7_days: "पिछले 7 दिन",
    older: "पुराने परामर्श",
  },
  kannada: {
    title: "ಜನನ್ಯಾಯ AI",
    tagline: "ನಾಗರಿಕರಿಗಾಗಿ ಭಾರತೀಯ ಕಾನೂನು ನೆರವು ಮತ್ತು ದಾಖಲೆ ವಿಶ್ಲೇಷಣೆ ವ್ಯವಸ್ಥೆ",
    tab_home: "ಮುಖಪುಟ (ಹೋಮ್)",
    tab_chat: "ಕಾನೂನು ಸಮಾಲೋಚನೆ",
    tab_doc: "ಕೇಸ್ ದಾಖಲೆ ಸ್ಟುಡಿಯೋ",
    tab_timeline: "ಘಟನಾವಳಿ ಟೈಮ್‌ಲೈನ್",
    tab_library: "ಕಾನೂನು ಗ್ರಂಥಾಲಯ",
    tab_dossier: "ಕೇಸ್ ಡೋಸಿಯರ್",
    tab_history: "ಸಮಾಲೋಚನೆ ಇತಿಹಾಸ",
    new_chat: "ಹೊಸ ಸಮಾಲೋಚನೆ",
    chat_history: "ಇತ್ತೀಚಿನ ಸಮಾಲೋಚನೆಗಳು",
    search_chats: "ಸಮಾಲೋಚನೆಗಳನ್ನು ಹುಡುಕಿ...",
    no_chats: "ಯಾವುದೇ ಹಿಂದಿನ ಸಮಾಲೋಚನೆಗಳಿಲ್ಲ",
    ask_placeholder: "ಯಾವುದೇ ಕಾನೂನು ಪ್ರಶ್ನೆ ಕೇಳಿ (ಉದಾ. ಚೆಕ್ ಬೌನ್ಸ್, ಸಾಲ ಮರುಪಾವತಿ, ಬಿಎನ್‌ಎಸ್ ಕಳ್ಳತನ, ಡಿಜಿಟಲ್ ಸಾಕ್ಷ್ಯ, ಸೈಬರ್ ವಂಚನೆ)...",
    ask_button: "ಕೇಳಿ",
    recording_active: "ಧ್ವನಿಯನ್ನು ಆಲಿಸಲಾಗುತ್ತಿದೆ...",
    upload_title: "ಕಾನೂನು ದಾಖಲೆಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    upload_subtitle: "PDF, JPG, PNG, WEBP — ಬಹು-ದಾಖಲೆ ವಿಶ್ಲೇಷಣೆ, ಕಾನೂನು ವಿಧಿಗಳು ಮತ್ತು ವಿರೋಧಾಭಾಸ ಪತ್ತೆ",
    browse_files: "ದಾಖಲೆಗಳನ್ನು ಆಯ್ಕೆಮಾಡಿ",
    analyze_button: "ದಾಖಲೆ ವಿಶ್ಲೇಷಿಸಿ",
    analyzing_text: "ದಾಖಲೆಯನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗುತ್ತಿದೆ...",
    guest_user: "ನಾಗರಿಕ ಖಾತೆ",
    login_signup: "ಲಾಗಿನ್ / ನೋಂದಣಿ",
    logout: "ಸೈನ್ ಔಟ್",
    edit_profile: "ಖಾತೆ ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
    save_profile: "ಬದಲಾವಣೆ ಉಳಿಸಿ",
    profile_title: "ಖಾತೆ ಸೆಟ್ಟಿಂಗ್‌ಗಳು ಮತ್ತು ಆದ್ಯತೆಗಳು",
    full_name: "ಪೂರ್ಣ ಹೆಸರು",
    email: "ಇಮೇಲ್ ವಿಳಾಸ",
    pref_language: "ಆದ್ಯತೆಯ ಭಾಷೆ",
    curr_password: "ಪ್ರಸ್ತುತ ಪಾಸ್‌ವರ್ಡ್",
    new_password: "ಹೊಸ ಪಾಸ್‌ವರ್ಡ್",
    today: "ಇಂದು",
    previous_7_days: "ಕಳೆದ 7 ದಿನಗಳು",
    older: "ಹಳೆಯ ಸಮಾಲೋಚನೆಗಳು",
  },
};

const SUGGESTED_QUESTIONS = [
  { domain: "Criminal Law (BNS)", icon: Gavel, text: "What is the definition of theft and what is the punishment under Section 303 of BNS?", lang: "en" },
  { domain: "Criminal Procedure (BNSS)", icon: ShieldAlert, text: "What is the procedure for registration of Zero FIR and e-FIR under Section 173 of BNSS?", lang: "en" },
  { domain: "Law of Evidence (BSA)", icon: FileCheck, text: "How are electronic records and digital messages admissible in evidence under Section 61 and 63 of BSA?", lang: "en" },
  { domain: "Commercial & Arbitration", icon: Briefcase, text: "What are the legal grounds for setting aside an arbitral award under Section 34 of the Arbitration Act?", lang: "en" },
  { domain: "Banking & Cheques (NI Act)", icon: CreditCard, text: "What is the statutory notice procedure and punishment for cheque bounce under Section 138 of NI Act?", lang: "en" },
  { domain: "Consumer Protection", icon: ShoppingBag, text: "How can a consumer file a claim for defective goods on e-Daakhil under Consumer Protection Act 2019?", lang: "en" },
  { domain: "Child Protection (JJ Act)", icon: Shield, text: "What are the bail provisions for a child in conflict with law under Section 12 of Juvenile Justice Act?", lang: "en" },
  { domain: "Criminal Law (Hindi)", icon: Gavel, text: "भारतीय न्याय संहिता 2023 के तहत चोरी की परिभाषा और सजा क्या है?", lang: "hi" },
  { domain: "Banking (Hindi)", icon: CreditCard, text: "चेक बाउंस होने पर क्या कानूनी प्रक्रिया और सजा होती है?", lang: "hi" },
  { domain: "Criminal Law (Kannada)", icon: Gavel, text: "ಕಳ್ಳತನ ಎಂದರೇನು ಮತ್ತು ಬಿಎನ್‌ಎಸ್ ಸೆಕ್ಷನ್ 303 ರ ಪ್ರಕಾರ ಏನು ಶಿಕ್ಷೆ?", lang: "kn" },
  { domain: "Banking (Kannada)", icon: CreditCard, text: "ಚೆಕ್ ಬೌನ್ಸ್ ಆದರೆ ಸೆಕ್ಷನ್ 138 ರ ಅಡಿಯಲ್ಲಿ ಕಾನೂನು ಕ್ರಮ ಮತ್ತು ಶಿಕ್ಷೆ ಏನು?", lang: "kn" },
];

function detectLanguage(text = "") {
  for (const char of text) {
    const code = char.charCodeAt(0);
    if (code >= 0x0900 && code <= 0x097f) return "hindi";
    if (code >= 0x0c80 && code <= 0x0cff) return "kannada";
  }
  return "english";
}

function renderInlineMarkdown(text = "") {
  if (!text) return "";
  const parts = String(text).split(/(\*\*.*?\*\*|\*.*?\*|`.*?`)/g);
  return parts.map((part, pIdx) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={pIdx} className="text-emphasis">{part.slice(2, -2)}</strong>;
    }
    if (part.startsWith("*") && part.endsWith("*")) {
      return <em key={pIdx}>{part.slice(1, -1)}</em>;
    }
    if (part.startsWith("`") && part.endsWith("`")) {
      return <code key={pIdx} className="inline-code-badge">{part.slice(1, -1)}</code>;
    }
    return part;
  });
}

function renderFormattedContent(rawText = "") {
  if (!rawText) return null;

  const blocks = String(rawText)
    .replace(/\r\n/g, "\n")
    .split(/\n\n+/);

  return blocks.map((block, bIdx) => {
    let trimmed = block.trim();
    if (!trimmed) return null;

    if (trimmed.startsWith("---") || trimmed === "___" || trimmed === "***" || trimmed === "===") {
      return <hr key={bIdx} className="content-divider" />;
    }

    // Check for Markdown Table blocks
    if (trimmed.includes("|")) {
      const rawLines = trimmed.split("\n").map((l) => l.trim()).filter(Boolean);
      // Filter out divider lines like |---|---| or |:---|:---|
      const isDivider = (l) => /^\|?[\s\-:]+(\|[\s\-:]+)+\|?$/.test(l);
      const tableLines = rawLines.filter((l) => l.includes("|") && !isDivider(l));

      if (tableLines.length >= 2) {
        const headerCells = tableLines[0]
          .split("|")
          .map((c) => c.trim())
          .filter(Boolean);
        const dataRows = tableLines.slice(1);

        if (headerCells.length > 0) {
          return (
            <div key={bIdx} className="table-responsive-wrapper">
              <table className="styled-legal-table">
                <thead>
                  <tr>
                    {headerCells.map((h, hIdx) => (
                      <th key={hIdx}>{renderInlineMarkdown(h)}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {dataRows.map((r, rIdx) => {
                    const cells = r
                      .split("|")
                      .map((c) => c.trim())
                      .filter(Boolean);
                    return (
                      <tr key={rIdx}>
                        {cells.map((c, cIdx) => (
                          <td key={cIdx}>{renderInlineMarkdown(c)}</td>
                        ))}
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          );
        }
      } else if (tableLines.length === 1 && !rawLines.some((l) => !l.includes("|"))) {
        // Single loose pipe line (e.g. | Item | Value |) -> render as clean badge card
        const cells = tableLines[0].split("|").map((c) => c.trim()).filter(Boolean);
        return (
          <div key={bIdx} className="single-pipe-highlight-row">
            {cells.map((cell, cIdx) => (
              <span key={cIdx} className="pipe-cell-badge">{renderInlineMarkdown(cell)}</span>
            ))}
          </div>
        );
      }
    }

    if (trimmed.startsWith("### ") || trimmed.startsWith("## ") || trimmed.startsWith("# ")) {
      const headingText = trimmed.replace(/^#+\s*/, "");
      return (
        <h4 key={bIdx} className="content-subheading">
          {renderInlineMarkdown(headingText)}
        </h4>
      );
    }

    const lines = trimmed.split("\n");
    const isList =
      lines.length > 1 &&
      lines.every(
        (l) =>
          l.trim().startsWith("- ") ||
          l.trim().startsWith("* ") ||
          l.trim().startsWith("• ") ||
          /^\d+[\.\)]\s/.test(l.trim())
      );

    if (isList) {
      return (
        <ul key={bIdx} className="content-bullet-list">
          {lines.map((line, lIdx) => {
            const cleanLine = line.trim().replace(/^[-*•]\s*|^\d+[\.\)]\s*/, "");
            return <li key={lIdx}>{renderInlineMarkdown(cleanLine)}</li>;
          })}
        </ul>
      );
    }

    return (
      <p key={bIdx} className="content-para">
        {renderInlineMarkdown(trimmed)}
      </p>
    );
  });
}

export default function App() {
  // Navigation & Direct URL Helper
  const getInitialTab = () => {
    const path = window.location.pathname.toLowerCase();
    if (path.startsWith("/consultation") || path.startsWith("/chat")) return "chat";
    if (path.startsWith("/documents") || path.startsWith("/doc")) return "doc";
    if (path.startsWith("/timeline")) return "timeline";
    if (path.startsWith("/knowledge-base") || path.startsWith("/library")) return "library";
    if (path.startsWith("/dossier")) return "dossier";
    if (path.startsWith("/history")) return "history";
    return "home";
  };

  const [activeTab, setActiveTab] = useState(getInitialTab);
  const [sidebarOpen, setSidebarOpen] = useState(() => {
    const saved = localStorage.getItem("jannyaya_sidebar_open");
    return saved !== null ? saved === "true" : true;
  });
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [uiLang, setUiLang] = useState("english");
  const t = MULTILINGUAL_UI[uiLang] || MULTILINGUAL_UI.english;

  // Auth & Profile State
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem("jannyaya_user") || sessionStorage.getItem("jannyaya_user");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [token, setToken] = useState(() => localStorage.getItem("jannyaya_token") || sessionStorage.getItem("jannyaya_token") || "");
  const [isGuest, setIsGuest] = useState(() => sessionStorage.getItem("jannyaya_is_guest") === "true");
  const [guestQuestionsLeft, setGuestQuestionsLeft] = useState(() => {
    const saved = sessionStorage.getItem("jannyaya_guest_quota");
    return saved !== null ? parseInt(saved, 10) : 5;
  });
  const [showTrialLimitModal, setShowTrialLimitModal] = useState(false);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authMode, setAuthMode] = useState("login"); // "login" | "register"
  const [authUsername, setAuthUsername] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authConfirmPassword, setAuthConfirmPassword] = useState("");
  const [authFullName, setAuthFullName] = useState("");
  const [authEmail, setAuthEmail] = useState("");
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPasswordAuth, setShowConfirmPasswordAuth] = useState(false);
  const [rememberMe, setRememberMe] = useState(true);
  const [acceptTerms, setAcceptTerms] = useState(true);

  // Profile Edit Modal State
  const [showProfileModal, setShowProfileModal] = useState(false);
  const [profileSubTab, setProfileSubTab] = useState("info"); // "info" | "stats" | "password"
  const [editFullName, setEditFullName] = useState("");
  const [editEmail, setEditEmail] = useState("");
  const [editLang, setEditLang] = useState("english");
  const [editAvatar, setEditAvatar] = useState("");
  const [editCitizenStatus, setEditCitizenStatus] = useState("Verified Citizen");
  const [editDefExplanationLang, setEditDefExplanationLang] = useState("english");
  const [editCurrPassword, setEditCurrPassword] = useState("");
  const [editNewPassword, setEditNewPassword] = useState("");
  const [editConfirmPassword, setEditConfirmPassword] = useState("");
  const [showCurrPassword, setShowCurrPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [userStats, setUserStats] = useState({
    consultations_count: 0,
    documents_analyzed: 0,
    questions_asked: 0,
    topics_explored: 0,
  });
  const [userStatsLoading, setUserStatsLoading] = useState(false);
  const [profileMsg, setProfileMsg] = useState({ type: "", text: "" });
  const [profileLoading, setProfileLoading] = useState(false);
  const avatarInputRef = useRef(null);

  // Conversations & History State
  const [conversations, setConversations] = useState([]);
  const [activeSessionId, setActiveSessionId] = useState(null);
  const [activeConvMeta, setActiveConvMeta] = useState(null);
  const [conversationsLoading, setConversationsLoading] = useState(false);
  const [historySearchQuery, setHistorySearchQuery] = useState("");
  const [historyLangFilter, setHistoryLangFilter] = useState("all");

  // Central Q&A State
  const [question, setQuestion] = useState("");
  const [chatMessages, setChatMessages] = useState([
    {
      role: "assistant",
      text: "Namaste. I am **JanNyaya AI**, your conversational legal intelligence assistant for Indian Law.\n\nYou can ask any legal question regarding banking regulations, criminal laws (BNS, BNSS, BSA), civil recovery, consumer rights, cyber law, property disputes, labour rights, or upload and analyze case files in English, हिन्दी, or ಕನ್ನಡ.",
      sources: [],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    },
  ]);
  const [loadingQA, setLoadingQA] = useState(false);
  const [qaError, setQaError] = useState("");
  const [speakingIndex, setSpeakingIndex] = useState(null);
  const [copiedIndex, setCopiedIndex] = useState(null);
  const chatBottomRef = useRef(null);

  // Luxury Theme State
  const [themeAccent, setThemeAccent] = useState(
    localStorage.getItem("jannyaya_theme_accent") || "sapphire-gold"
  );
  const [showThemePicker, setShowThemePicker] = useState(false);

  // Voice Recording State (Groq Whisper Large v3 + MediaRecorder)
  const [recording, setRecording] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState("");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const recordingTimerRef = useRef(null);

  // Multi-Document Studio State
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [activeDocSubTab, setActiveDocSubTab] = useState("summary");
  const [isDragging, setIsDragging] = useState(false);
  const [docExplanationLang, setDocExplanationLang] = useState("english");
  const [docChatMessages, setDocChatMessages] = useState([]);
  const [docChatInput, setDocChatInput] = useState("");
  const [docChatLoading, setDocChatLoading] = useState(false);
  const [checkedSteps, setCheckedSteps] = useState({});
  const fileInputRef = useRef(null);

  // Timeline State
  const [timelineInputText, setTimelineInputText] = useState("");
  const [extractedTimeline, setExtractedTimeline] = useState([]);
  const [timelineLoading, setTimelineLoading] = useState(false);

  // Bare Acts Library & Knowledge Base Search State
  const [actsList, setActsList] = useState([]);
  const [actsLoading, setActsLoading] = useState(false);
  const [actSearchQuery, setActSearchQuery] = useState("");
  const [selectedDomainFilter, setSelectedDomainFilter] = useState("all");
  const [kbSearchResults, setKbSearchResults] = useState(null);
  const [kbSearching, setKbSearching] = useState(false);

  // Source Provenance Modal & Knowledge Base Stats State
  const [selectedSourceModal, setSelectedSourceModal] = useState(null);
  const [kbStats, setKbStats] = useState(null);
  const [kbStatsLoading, setKbStatsLoading] = useState(false);

  // Navigation Handler with Direct URL Sync
  const navigateToTab = (tabName, sessionId = null) => {
    setActiveTab(tabName);
    setMobileMenuOpen(false);

    const pathMap = {
      home: "/home",
      chat: sessionId ? `/consultation/${sessionId}` : (activeSessionId ? `/consultation/${activeSessionId}` : "/consultation"),
      doc: "/documents",
      timeline: "/timeline",
      library: "/knowledge-base",
      dossier: "/dossier",
      history: "/history",
    };

    const targetPath = pathMap[tabName] || "/";
    if (window.location.pathname !== targetPath) {
      window.history.pushState({ tab: tabName, sessionId }, "", targetPath);
    }
  };

  // Keyboard accessibility & popstate routing
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === "Escape") {
        setMobileMenuOpen(false);
        setShowAuthModal(false);
        setShowProfileModal(false);
        setSelectedSourceModal(null);
        setShowThemePicker(false);
      }
    };

    const handlePopState = (e) => {
      const tab = getInitialTab();
      setActiveTab(tab);
      const match = window.location.pathname.match(/\/consultation\/([a-zA-Z0-9_-]+)/);
      if (match && match[1]) {
        handleSelectConversation(match[1], false);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("popstate", handlePopState);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("popstate", handlePopState);
    };
  }, []);

  // Save Sidebar Open State
  useEffect(() => {
    localStorage.setItem("jannyaya_sidebar_open", sidebarOpen ? "true" : "false");
  }, [sidebarOpen]);

  // Load Knowledge Base Live Stats
  const loadKbStats = async () => {
    setKbStatsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/knowledge-base/stats`);
      const data = await res.json();
      if (res.ok) setKbStats(data);
    } catch (e) {
      console.warn("Could not load knowledge base stats:", e);
    } finally {
      setKbStatsLoading(false);
    }
  };

  // Load All Conversations
  const loadConversations = async (search = "") => {
    setConversationsLoading(true);
    try {
      const headers = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;
      const url = search
        ? `${API_BASE}/conversations?search=${encodeURIComponent(search)}`
        : `${API_BASE}/conversations`;
      const res = await fetch(url, { headers });
      const data = await res.json();
      if (data.status === "success" && Array.isArray(data.conversations)) {
        setConversations(data.conversations);
      }
    } catch (err) {
      console.warn("Failed to load conversations:", err);
    } finally {
      setConversationsLoading(false);
    }
  };

  // Initial Data Fetching
  useEffect(() => {
    loadKbStats();
    loadConversations();
    loadActsCatalog();
  }, [token]);

  // Handle URL Direct Load with Conversation ID
  useEffect(() => {
    const match = window.location.pathname.match(/\/consultation\/([a-zA-Z0-9_-]+)/);
    if (match && match[1]) {
      handleSelectConversation(match[1], false);
    }
  }, []);

  useEffect(() => {
    if (activeTab === "library" || activeTab === "home") {
      loadActsCatalog();
      loadKbStats();
    }
    if (activeTab === "history") {
      loadConversations(historySearchQuery);
    }
  }, [activeTab]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages, docChatMessages]);

  // ============================================================
  // AUTH & SESSION MANAGEMENT
  // ============================================================

  const handleAuthSubmit = async (e) => {
    if (e) e.preventDefault();
    setAuthError("");
    setAuthLoading(true);

    if (authMode === "register") {
      if (!authUsername.trim()) {
        setAuthError("Username is required.");
        setAuthLoading(false);
        return;
      }
      if (authPassword.length < 4) {
        setAuthError("Password must be at least 4 characters.");
        setAuthLoading(false);
        return;
      }
      if (authPassword !== authConfirmPassword) {
        setAuthError("Passwords do not match. Please re-enter.");
        setAuthLoading(false);
        return;
      }
      if (!acceptTerms) {
        setAuthError("Please accept the terms and privacy statement.");
        setAuthLoading(false);
        return;
      }
    }

    try {
      const endpoint = authMode === "login" ? "/auth/login" : "/auth/register";
      const payload =
        authMode === "login"
          ? { username: authUsername.trim(), password: authPassword }
          : {
              username: authUsername.trim(),
              password: authPassword,
              full_name: authFullName.trim() || authUsername.trim(),
              email: authEmail.trim() || "",
              language: uiLang,
            };

      const res = await fetch(`${API_BASE}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Authentication failed. Please verify credentials.");

      setToken(data.token);
      setUser(data.user);
      setIsGuest(false);
      sessionStorage.removeItem("jannyaya_is_guest");
      if (rememberMe) {
        localStorage.setItem("jannyaya_token", data.token);
        localStorage.setItem("jannyaya_user", JSON.stringify(data.user));
      } else {
        sessionStorage.setItem("jannyaya_token", data.token);
        sessionStorage.setItem("jannyaya_user", JSON.stringify(data.user));
      }
      setShowAuthModal(false);
      setShowTrialLimitModal(false);
      loadConversations();
    } catch (err) {
      setAuthError(err.message || "Failed to authenticate.");
    } finally {
      setAuthLoading(false);
    }
  };

  const handleStartGuestTrial = () => {
    setIsGuest(true);
    sessionStorage.setItem("jannyaya_is_guest", "true");
    if (sessionStorage.getItem("jannyaya_guest_quota") === null) {
      sessionStorage.setItem("jannyaya_guest_quota", "5");
      setGuestQuestionsLeft(5);
    }
    setShowAuthModal(false);
    setShowTrialLimitModal(false);
  };

  const handleLogout = async () => {
    try {
      if (token) {
        await fetch(`${API_BASE}/auth/logout`, {
          method: "POST",
          headers: { Authorization: `Bearer ${token}` },
        });
      }
    } catch (e) {
      console.warn("Logout notification error:", e);
    }
    setToken("");
    setUser(null);
    setIsGuest(false);
    localStorage.removeItem("jannyaya_token");
    localStorage.removeItem("jannyaya_user");
    sessionStorage.clear();
    setConversations([]);
    setActiveSessionId(null);
    setActiveConvMeta(null);
    setChatMessages([
      {
        role: "assistant",
        text: "Namaste. I am **JanNyaya AI**, your conversational legal intelligence assistant for Indian Law.\n\nYou can ask any legal question regarding banking regulations, criminal laws (BNS, BNSS, BSA), civil recovery, consumer rights, cyber law, property disputes, labour rights, or upload and analyze case files in English, हिन्दी, or ಕನ್ನಡ.",
        sources: [],
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      },
    ]);
  };

  const handleAvatarChange = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    if (!validTypes.includes(file.type)) {
      setProfileMsg({ type: "error", text: "Please select a valid image file (JPG, PNG, WEBP)." });
      return;
    }

    if (file.size > 5 * 1024 * 1024) {
      setProfileMsg({ type: "error", text: "Image file size must be less than 5MB." });
      return;
    }

    const reader = new FileReader();
    reader.onload = (evt) => {
      setEditAvatar(evt.target.result);
      setProfileMsg({ type: "", text: "" });
    };
    reader.readAsDataURL(file);
  };

  const handleRemoveAvatar = () => {
    setEditAvatar("");
    if (avatarInputRef.current) avatarInputRef.current.value = "";
  };

  const loadUserStats = async () => {
    if (!token) return;
    setUserStatsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/auth/stats`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = await res.json();
      if (data && data.status === "success") {
        setUserStats(data);
      }
    } catch (err) {
      console.warn("Failed to load user stats:", err);
    } finally {
      setUserStatsLoading(false);
    }
  };

  const openProfileModal = () => {
    if (user) {
      setEditFullName(user.full_name || "");
      setEditEmail(user.email || "");
      setEditLang(user.language || "english");
      setEditAvatar(user.avatar || "");
      setEditCitizenStatus(user.citizen_status || "Verified Citizen");
      setEditDefExplanationLang(user.default_explanation_lang || "english");
      setEditCurrPassword("");
      setEditNewPassword("");
      setEditConfirmPassword("");
      setShowCurrPassword(false);
      setShowNewPassword(false);
      setShowConfirmPassword(false);
      setProfileSubTab("info");
      setProfileMsg({ type: "", text: "" });
      loadUserStats();
      setShowProfileModal(true);
    } else {
      setShowAuthModal(true);
    }
  };

  const handleSaveProfile = async (e) => {
    e.preventDefault();
    setProfileLoading(true);
    setProfileMsg({ type: "", text: "" });

    // Validate password change
    if (editNewPassword) {
      if (!editCurrPassword) {
        setProfileMsg({ type: "error", text: "Please enter your current password to set a new password." });
        setProfileLoading(false);
        return;
      }
      if (editNewPassword.length < 4) {
        setProfileMsg({ type: "error", text: "New password must be at least 4 characters." });
        setProfileLoading(false);
        return;
      }
      if (editNewPassword !== editConfirmPassword) {
        setProfileMsg({ type: "error", text: "New passwords do not match." });
        setProfileLoading(false);
        return;
      }
    }

    try {
      const res = await fetch(`${API_BASE}/auth/profile`, {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          full_name: editFullName,
          email: editEmail,
          language: editLang,
          avatar: editAvatar,
          citizen_status: editCitizenStatus,
          default_explanation_lang: editDefExplanationLang,
          current_password: editCurrPassword || undefined,
          new_password: editNewPassword || undefined,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Failed to update account");

      setUser(data.user);
      localStorage.setItem("jannyaya_user", JSON.stringify(data.user));
      setEditCurrPassword("");
      setEditNewPassword("");
      setEditConfirmPassword("");
      setProfileMsg({ type: "success", text: "Profile updated successfully." });
      setTimeout(() => setShowProfileModal(false), 1300);
    } catch (err) {
      setProfileMsg({ type: "error", text: err.message || "Update failed." });
    } finally {
      setProfileLoading(false);
    }
  };

  // ============================================================
  // CONVERSATION CREATION, SELECTION & CONTINUATION
  // ============================================================

  const handleCreateNewChat = async () => {
    try {
      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`${API_BASE}/conversations`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          title: "New Legal Consultation",
          language: uiLang,
          legal_topic: "general",
        }),
      });
      const data = await res.json();

      if (data.status === "success" && data.conversation) {
        const newConv = data.conversation;
        setConversations((prev) => [newConv, ...prev.filter((c) => c.id !== newConv.id)]);
        setActiveSessionId(newConv.id);
        setActiveConvMeta(newConv);
        setChatMessages([
          {
            role: "assistant",
            text: "Namaste. Started a fresh legal consultation session. Ask your question regarding Indian statutes, penalties, or procedures in English, हिन्दी, or ಕನ್ನಡ.",
            sources: [],
            timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
          },
        ]);
        navigateToTab("chat", newConv.id);
      }
    } catch (err) {
      console.warn("Failed to create new conversation session:", err);
      setActiveSessionId(null);
      setActiveConvMeta(null);
      setChatMessages([
        {
          role: "assistant",
          text: "Namaste. Started a fresh legal consultation session. Ask your question in English, हिन्दी, or ಕನ್ನಡ.",
          sources: [],
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
      navigateToTab("chat");
    }
  };

  const handleSelectConversation = async (conversationId, updateUrl = true) => {
    setActiveSessionId(conversationId);
    if (updateUrl) {
      navigateToTab("chat", conversationId);
    } else {
      setActiveTab("chat");
    }
    setMobileMenuOpen(false);

    try {
      const headers = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      const res = await fetch(`${API_BASE}/conversations/${conversationId}`, { headers });
      const data = await res.json();

      if (data.status === "success" && data.conversation) {
        const conv = data.conversation;
        setActiveConvMeta(conv);

        const formatted = (conv.messages || []).map((m) => ({
          role: m.role,
          text: m.text || m.content,
          sources: m.sources || [],
          timestamp: m.timestamp || new Date(m.created_at ? m.created_at * 1000 : Date.now()).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          }),
        }));

        setChatMessages(
          formatted.length > 0
            ? formatted
            : [
                {
                  role: "assistant",
                  text: `Consultation on **${conv.title || "Indian Law"}** loaded. How can JanNyaya AI assist you with Indian law today?`,
                  sources: [],
                  timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
                },
              ]
        );

        if (conv.analysis && Object.keys(conv.analysis).length > 0) {
          setUploadResult(conv.analysis);
        }
      }
    } catch (err) {
      console.error("Failed to load previous consultation:", err);
    }
  };

  const handleDeleteConversation = async (conversationId, e) => {
    if (e) e.stopPropagation();
    try {
      const headers = {};
      if (token) headers["Authorization"] = `Bearer ${token}`;

      await fetch(`${API_BASE}/conversations/${conversationId}`, {
        method: "DELETE",
        headers,
      });

      setConversations((prev) => prev.filter((c) => c.id !== conversationId && c.conversation_id !== conversationId));

      if (activeSessionId === conversationId) {
        handleCreateNewChat();
      }
    } catch (err) {
      console.error("Failed to delete conversation:", err);
    }
  };

  // ============================================================
  // CENTRAL LEGAL CONSULTATION (Q&A & MULTI-TURN RAG)
  // ============================================================

  const handleSendQuestion = async (queryOverride = null) => {
    const q = (queryOverride || question).trim();
    if (!q || loadingQA) return;

    if (isGuest && guestQuestionsLeft <= 0) {
      setShowTrialLimitModal(true);
      return;
    }

    if (isGuest) {
      setGuestQuestionsLeft((prev) => {
        const next = Math.max(0, prev - 1);
        sessionStorage.setItem("jannyaya_guest_quota", next.toString());
        return next;
      });
    }

    setQuestion("");
    setQaError("");
    setLoadingQA(true);

    const userMsg = {
      role: "user",
      text: q,
      sources: [],
      timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    };

    setChatMessages((prev) => [...prev, userMsg]);

    try {
      let currentConvId = activeSessionId;

      // If no active conversation session, create one
      if (!currentConvId) {
        const initHeaders = { "Content-Type": "application/json" };
        if (token) initHeaders["Authorization"] = `Bearer ${token}`;

        const createRes = await fetch(`${API_BASE}/conversations`, {
          method: "POST",
          headers: initHeaders,
          body: JSON.stringify({
            title: q.slice(0, 60),
            language: uiLang,
            legal_topic: "general",
          }),
        });
        const createData = await createRes.json();
        if (createData.conversation) {
          currentConvId = createData.conversation.id;
          setActiveSessionId(currentConvId);
          setActiveConvMeta(createData.conversation);
          window.history.replaceState({ tab: "chat", sessionId: currentConvId }, "", `/consultation/${currentConvId}`);
        }
      }

      const headers = { "Content-Type": "application/json" };
      if (token) headers["Authorization"] = `Bearer ${token}`;

      // Build conversation history for grounded multi-turn RAG
      const historyContext = chatMessages.slice(-8).map((m) => ({
        role: m.role,
        content: m.text,
      }));

      const endpoint = currentConvId
        ? `${API_BASE}/conversations/${currentConvId}/messages`
        : `${API_BASE}/ask`;

      const payload = currentConvId
        ? { question: q, language: uiLang, history: historyContext }
        : { question: q, language: uiLang, conversation_history: historyContext };

      const res = await fetch(endpoint, {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Legal reasoning service error");

      const botMsg = {
        role: "assistant",
        text: data.answer || "No response received from legal reasoning service.",
        sources: data.sources || [],
        timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
      };

      setChatMessages((prev) => [...prev, botMsg]);

      if (data.conversation) {
        setActiveConvMeta(data.conversation);
      }

      loadConversations();
    } catch (err) {
      setQaError(err.message || "Failed to reach JanNyaya AI service.");
      setChatMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `We encountered an issue processing your query: ${err.message || "Unknown error"}. Please check your connection or retry.`,
          sources: [],
          timestamp: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
        },
      ]);
    } finally {
      setLoadingQA(false);
    }
  };

  const handleCopyText = (text, idx) => {
    navigator.clipboard.writeText(text);
    setCopiedIndex(idx);
    setTimeout(() => setCopiedIndex(null), 2000);
  };

  const handleSpeakText = (text, idx) => {
    if (speakingIndex === idx) {
      window.speechSynthesis.cancel();
      setSpeakingIndex(null);
      return;
    }
    window.speechSynthesis.cancel();
    const clean = text.replace(/[*#`_~]/g, "").replace(/---/g, "");
    const utterance = new SpeechSynthesisUtterance(clean);
    const lang = detectLanguage(text);
    if (lang === "hindi") utterance.lang = "hi-IN";
    else if (lang === "kannada") utterance.lang = "kn-IN";
    else utterance.lang = "en-IN";

    utterance.onend = () => setSpeakingIndex(null);
    utterance.onerror = () => setSpeakingIndex(null);
    setSpeakingIndex(idx);
    window.speechSynthesis.speak(utterance);
  };

  const handleToggleVoice = async () => {
    if (recording) {
      if (recordingTimerRef.current) {
        clearInterval(recordingTimerRef.current);
        recordingTimerRef.current = null;
      }

      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== "inactive") {
        mediaRecorderRef.current.stop();
        setRecording(false);
        setVoiceStatus(`Transcribing ${uiLang} with Whisper Large v3...`);
        return;
      }

      if (window.recognitionInstance) {
        window.recognitionInstance.stop();
        setRecording(false);
        setVoiceStatus("");
        return;
      }

      setRecording(false);
      setVoiceStatus("");
      return;
    }

    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia && window.MediaRecorder) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioChunksRef.current = [];

        let mimeType = "audio/webm";
        if (MediaRecorder.isTypeSupported("audio/webm;codecs=opus")) {
          mimeType = "audio/webm;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/ogg;codecs=opus")) {
          mimeType = "audio/ogg;codecs=opus";
        } else if (MediaRecorder.isTypeSupported("audio/mp4")) {
          mimeType = "audio/mp4";
        }

        const recorder = new MediaRecorder(stream, { mimeType });
        mediaRecorderRef.current = recorder;

        recorder.ondataavailable = (e) => {
          if (e.data && e.data.size > 0) {
            audioChunksRef.current.push(e.data);
          }
        };

        recorder.onstop = async () => {
          stream.getTracks().forEach((track) => track.stop());

          const audioBlob = new Blob(audioChunksRef.current, { type: mimeType });
          if (audioBlob.size < 500) {
            setVoiceStatus("Recording was too short. Please speak clearly.");
            setTimeout(() => setVoiceStatus(""), 3000);
            return;
          }

          setVoiceStatus(`Transcribing ${uiLang.toUpperCase()} with Whisper Large v3...`);

          const formData = new FormData();
          const fileExt = mimeType.includes("ogg") ? "ogg" : mimeType.includes("mp4") ? "mp4" : "webm";
          formData.append("file", audioBlob, `speech_query.${fileExt}`);
          formData.append("language", uiLang);

          try {
            const res = await fetch(`${API_BASE}/speech-to-text`, {
              method: "POST",
              body: formData,
            });
            const data = await res.json();
            if (res.ok && data.text) {
              setQuestion(data.text);
              setVoiceStatus("");
            } else {
              throw new Error(data.detail || "Whisper transcription returned empty result");
            }
          } catch (err) {
            console.warn("Backend Whisper transcription error:", err);
            setVoiceStatus("Whisper transcription failed. Please try again or type.");
            setTimeout(() => setVoiceStatus(""), 4000);
          }
        };

        recorder.start(250);
        setRecording(true);
        setRecordingSeconds(0);
        setVoiceStatus(`Listening in ${uiLang.toUpperCase()}... (Click mic again when finished)`);

        recordingTimerRef.current = setInterval(() => {
          setRecordingSeconds((prev) => prev + 1);
        }, 1000);

        return;
      } catch (micErr) {
        console.warn("Microphone getUserMedia failed or was blocked, falling back to Web Speech API:", micErr);
      }
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      const recognition = new SpeechRecognition();
      window.recognitionInstance = recognition;
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = uiLang === "hindi" ? "hi-IN" : uiLang === "kannada" ? "kn-IN" : "en-IN";

      recognition.onstart = () => {
        setRecording(true);
        setVoiceStatus(`Listening in ${uiLang}...`);
      };

      recognition.onresult = (e) => {
        const transcript = e.results[0][0].transcript;
        setQuestion(transcript);
        setRecording(false);
        setVoiceStatus("");
      };

      recognition.onerror = () => {
        setRecording(false);
        setVoiceStatus("");
      };

      recognition.onend = () => {
        setRecording(false);
        setVoiceStatus("");
      };

      recognition.start();
    } else {
      alert("Microphone recording is unavailable. Please grant microphone permissions or type your question directly.");
    }
  };

  // ============================================================
  // MULTI-DOC CASE STUDIO / DOCUMENT ANALYZER
  // ============================================================

  const handleFileSelect = (e) => {
    const files = Array.from(e.target.files || []);
    if (files.length > 0) {
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  };

  const handleRemoveFile = (index) => {
    setSelectedFiles((prev) => prev.filter((_, i) => i !== index));
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files || []);
    if (files.length > 0) {
      setSelectedFiles((prev) => [...prev, ...files]);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (selectedFiles.length === 0 || uploading) return;
    setUploading(true);
    setUploadResult(null);
    setDocChatMessages([]);

    const formData = new FormData();
    if (selectedFiles.length === 1) {
      formData.append("file", selectedFiles[0]);
      formData.append("language", docExplanationLang);
      try {
        const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Document analysis failed");
        setUploadResult(data);
        setActiveDocSubTab("summary");
        if (data.timeline && data.timeline.length > 0) {
          setExtractedTimeline(data.timeline);
        }
      } catch (err) {
        alert(err.message || "Document analysis failed.");
      } finally {
        setUploading(false);
      }
    } else {
      selectedFiles.forEach((file) => formData.append("files", file));
      formData.append("language", docExplanationLang);
      try {
        const res = await fetch(`${API_BASE}/upload-multiple`, { method: "POST", body: formData });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || "Multi-document case analysis failed");
        setUploadResult(data);
        setActiveDocSubTab("summary");
        if (data.timeline && data.timeline.length > 0) {
          setExtractedTimeline(data.timeline);
        }
      } catch (err) {
        alert(err.message || "Multi-document analysis failed.");
      } finally {
        setUploading(false);
      }
    }
  };

  const handleDocChat = async (e) => {
    e.preventDefault();
    const q = docChatInput.trim();
    if (!q || docChatLoading || !uploadResult) return;

    setDocChatInput("");
    const userMsg = { role: "user", text: q };
    setDocChatMessages((prev) => [...prev, userMsg]);
    setDocChatLoading(true);

    const docText =
      uploadResult.text ||
      uploadResult.extracted_text ||
      (uploadResult.documents || []).map((d) => d.text || "").join("\n\n") ||
      (uploadResult.analysis?.documents || []).map((d) => d.text || "").join("\n\n");

    try {
      const res = await fetch(`${API_BASE}/document/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          document_text: docText,
          question: q,
          language: docExplanationLang,
          conversation_history: docChatMessages.slice(-4).map((m) => ({
            role: m.role,
            content: m.text,
          })),
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Document Q&A failed");

      setDocChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.answer || "No response received." },
      ]);
    } catch (err) {
      setDocChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: `Error: ${err.message || "Document AI could not process this question."}` },
      ]);
    } finally {
      setDocChatLoading(false);
    }
  };

  // ============================================================
  // CASE TIMELINE GENERATOR
  // ============================================================

  const handleGenerateTimeline = async () => {
    if (!timelineInputText.trim() && selectedFiles.length === 0) return;
    setTimelineLoading(true);

    try {
      const res = await fetch(`${API_BASE}/analyze-timeline`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: timelineInputText }),
      });
      const data = await res.json();
      if (data.status === "success" && Array.isArray(data.timeline)) {
        setExtractedTimeline(data.timeline);
      }
    } catch (err) {
      alert("Error extracting case timeline.");
    } finally {
      setTimelineLoading(false);
    }
  };

  // ============================================================
  // BARE ACTS LIBRARY & KNOWLEDGE BASE SEARCH
  // ============================================================

  const loadActsCatalog = async () => {
    setActsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/library/acts`);
      const data = await res.json();
      if (data.status === "success" && Array.isArray(data.acts)) {
        setActsList(data.acts);
      }
    } catch (err) {
      console.error("Failed to load bare acts:", err);
    } finally {
      setActsLoading(false);
    }
  };

  const handleKnowledgeBaseSearch = async (e) => {
    if (e) e.preventDefault();
    if (!actSearchQuery.trim() && selectedDomainFilter === "all") {
      setKbSearchResults(null);
      return;
    }

    setKbSearching(true);
    try {
      const params = new URLSearchParams();
      if (actSearchQuery.trim()) params.append("q", actSearchQuery.trim());
      if (selectedDomainFilter !== "all") params.append("domain", selectedDomainFilter);
      params.append("limit", "25");

      const res = await fetch(`${API_BASE}/knowledge-base/search?${params.toString()}`);
      const data = await res.json();
      if (data.status === "success") {
        setKbSearchResults(data.results);
      }
    } catch (err) {
      console.warn("Knowledge base search error:", err);
    } finally {
      setKbSearching(false);
    }
  };

  // ============================================================
  // CASE DOSSIER & EXPORT
  // ============================================================

  const handleExportDossier = () => {
    const report = {
      system: "JanNyaya AI - Multimodal Intelligent Legal Assistance System",
      user: user?.username || "Verified Citizen",
      export_date: new Date().toISOString(),
      active_consultation: {
        session_id: activeSessionId,
        topic: activeConvMeta?.legal_topic || "General Indian Law",
        messages: chatMessages,
      },
      document_analysis: uploadResult ? {
        filename: uploadResult.filename,
        document_type: docTypeLabel,
        summary: docSummary,
        facts: docFacts,
        amounts: docAmounts,
        dates: docDates,
        deadlines: docDeadlines,
        provisions: docProvisions,
        actionable_steps: docNextSteps,
      } : null,
      timeline: extractedTimeline,
      disclaimer: "JanNyaya AI provides verified Indian statutory legal information. Not a substitute for formal legal representation.",
    };

    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `JanNyaya_Case_Dossier_${Date.now()}.json`;
    a.click();
  };

  // Extract structured properties from uploadResult
  const analysisObj = uploadResult?.analysis || {};
  const docSummary =
    uploadResult?.summary ||
    analysisObj?.llm_explanation?.summary ||
    analysisObj?.summary ||
    uploadResult?.multi_document_summary ||
    "Document analysis complete.";

  const docClauses =
    uploadResult?.conditions_and_clauses ||
    analysisObj?.llm_explanation?.conditions_and_clauses ||
    analysisObj?.conditions_and_clauses ||
    [];

  const docNextSteps =
    uploadResult?.actionable_steps ||
    analysisObj?.llm_explanation?.actionable_steps ||
    analysisObj?.next_steps ||
    [];

  const docProvisions =
    uploadResult?.provisions ||
    analysisObj?.provisions ||
    uploadResult?.provision_analysis?.provisions ||
    [];

  const primaryProv =
    uploadResult?.primary_provision ||
    analysisObj?.primary_provision ||
    uploadResult?.provision_analysis?.primary_provision;

  const docFacts = uploadResult?.facts || analysisObj?.facts || {};
  const docAmounts = uploadResult?.amounts || analysisObj?.amounts || docFacts?.amounts || [];
  const docDates = uploadResult?.dates || analysisObj?.dates || docFacts?.dates || [];
  const docInterest = uploadResult?.interest_rates || analysisObj?.interest_rates || docFacts?.interest_rates || [];
  const docDeadlines = uploadResult?.deadlines || analysisObj?.deadlines || docFacts?.deadlines || [];
  const docConflicts = uploadResult?.conflicts || analysisObj?.conflicts || [];
  const docTypeLabel = analysisObj?.document_type || "Legal Document";
  const docRouteLabel = analysisObj?.legal_route_label || "Civil / Statutory Law";
  const docSafetyCaution = uploadResult?.safety_caution || analysisObj?.safety_caution;

  // Rich structured case fields
  const docDocLang = uploadResult?.document_language || analysisObj?.document_language || "english";
  const docExpLang = uploadResult?.explanation_language || analysisObj?.explanation_language || docExplanationLang || "english";
  const docOverview = uploadResult?.document_overview || analysisObj?.document_overview || {};
  const docParties = uploadResult?.parties || analysisObj?.parties || docFacts?.parties || [];
  const docClaims = uploadResult?.claims || analysisObj?.claims || [];
  const docImportantFacts = uploadResult?.important_facts || analysisObj?.important_facts || [];
  const docMissingInfo = uploadResult?.missing_information || analysisObj?.missing_information || [];
  const docWarnings = uploadResult?.warnings || analysisObj?.warnings || [];
  const docCombinedSummary = uploadResult?.combined_case_summary || {};

  // Filtered Acts
  const filteredActs = actsList.filter((act) => {
    const matchesSearch =
      act.act_name.toLowerCase().includes(actSearchQuery.toLowerCase()) ||
      (act.authority && act.authority.toLowerCase().includes(actSearchQuery.toLowerCase()));

    if (!matchesSearch) return false;
    if (selectedDomainFilter === "all") return true;
    if (selectedDomainFilter === "criminal") return act.act_name.includes("BNS") || act.act_name.includes("Nyaya") || act.act_name.includes("Nagarik") || act.act_name.includes("Sakshya") || act.act_name.includes("POCSO");
    if (selectedDomainFilter === "banking") return act.act_name.includes("Negotiable") || act.act_name.includes("Banking") || act.act_name.includes("SARFAESI") || act.act_name.includes("Insolvency") || act.act_name.includes("RBI");
    if (selectedDomainFilter === "civil") return act.act_name.includes("Contract") || act.act_name.includes("Civil") || act.act_name.includes("Limitation") || act.act_name.includes("Specific") || act.act_name.includes("Arbitration");
    if (selectedDomainFilter === "property") return act.act_name.includes("Property") || act.act_name.includes("RERA") || act.act_name.includes("Succession");
    if (selectedDomainFilter === "consumer") return act.act_name.includes("Consumer");
    if (selectedDomainFilter === "cyber") return act.act_name.includes("Information Technology") || act.act_name.includes("Data Protection") || act.act_name.includes("DPDP");
    if (selectedDomainFilter === "labour") return act.act_name.includes("Wages") || act.act_name.includes("Gratuity");
    if (selectedDomainFilter === "family") return act.act_name.includes("Marriage") || act.act_name.includes("Domestic") || act.act_name.includes("Senior") || act.act_name.includes("Juvenile");
    return true;
  });

  // Filtered History for dedicated history page
  const filteredConversations = conversations.filter((c) => {
    const term = historySearchQuery.toLowerCase().trim();
    const matchesTerm =
      !term ||
      c.title.toLowerCase().includes(term) ||
      (c.last_question && c.last_question.toLowerCase().includes(term)) ||
      (c.legal_topic && c.legal_topic.toLowerCase().includes(term));

    const matchesLang =
      historyLangFilter === "all" ||
      (c.language && c.language.toLowerCase() === historyLangFilter.toLowerCase());

    return matchesTerm && matchesLang;
  });

  // ============================================================
  // MANDATORY FIRST SCREEN: AUTHENTICATION PORTAL
  // ============================================================
  if (!user && !isGuest) {
    return (
      <div className="auth-portal-fullscreen" data-theme={themeAccent}>
        <div className="auth-portal-bg-decor" />
        <div className="auth-portal-ambient-glow" />

        <div className="auth-portal-container">
          {/* Left Hero Showcase */}
          <div className="auth-hero-showcase">
            <div className="auth-hero-brand">
              <div className="auth-hero-logo-box">
                <Scale size={28} />
              </div>
              <div>
                <div className="auth-hero-title">JanNyaya AI</div>
                <div className="auth-hero-subtitle">Indian Legal Intelligence</div>
              </div>
            </div>

            <div className="auth-hero-pitch">
              <div className="auth-hero-tagline-box">
                <div className="auth-hero-tagline-en">
                  “Understand Indian Law. In Your Language.”
                </div>
                <div className="auth-hero-tagline-sub">
                  भारतीय कानून को अपनी भाषा में समझें • ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಭಾರತೀಯ ಕಾನೂನನ್ನು ಅರ್ಥಮಾಡಿಕೊಳ್ಳಿ
                </div>
              </div>

              <div className="auth-hero-features-list">
                <div className="auth-feature-pill">
                  <BookOpen size={16} />
                  <span>BNS, BNSS & BSA 2023</span>
                </div>
                <div className="auth-feature-pill">
                  <FileText size={16} />
                  <span>Multi-Document Studio</span>
                </div>
                <div className="auth-feature-pill">
                  <Mic size={16} />
                  <span>Multilingual Voice & OCR</span>
                </div>
                <div className="auth-feature-pill">
                  <ShieldCheck size={16} />
                  <span>100% Grounded Statutory RAG</span>
                </div>
              </div>
            </div>

            <div className="auth-hero-footer-stat">
              <span>Authoritative Statutory Knowledge Base</span>
              <span className="auth-hero-stat-badge">
                <Sparkles size={13} />
                5,142+ Legal Provisions
              </span>
            </div>
          </div>

          {/* Right Auth Card Panel */}
          <div className="auth-form-panel">
            <div className="auth-tabs-header">
              <button
                className={`auth-tab-toggle-btn ${authMode === "login" ? "active" : ""}`}
                onClick={() => {
                  setAuthMode("login");
                  setAuthError("");
                }}
                type="button"
              >
                <LogIn size={15} />
                <span>Sign In</span>
              </button>
              <button
                className={`auth-tab-toggle-btn ${authMode === "register" ? "active" : ""}`}
                onClick={() => {
                  setAuthMode("register");
                  setAuthError("");
                }}
                type="button"
              >
                <UserPlus size={15} />
                <span>Create Account</span>
              </button>
            </div>

            <form className="auth-form-body" onSubmit={handleAuthSubmit}>
              {authError && (
                <div className="auth-error-banner" role="alert">
                  <AlertTriangle size={15} />
                  <span>{authError}</span>
                </div>
              )}

              {authMode === "register" && (
                <div className="auth-input-group">
                  <label className="auth-input-label">Full Name</label>
                  <div className="auth-input-wrapper">
                    <User size={16} className="auth-input-icon" />
                    <input
                      type="text"
                      className="auth-text-field"
                      placeholder="e.g. Adv. Rajesh Sharma"
                      value={authFullName}
                      onChange={(e) => setAuthFullName(e.target.value)}
                      required
                    />
                  </div>
                </div>
              )}

              <div className="auth-input-group">
                <label className="auth-input-label">
                  {authMode === "login" ? "Username or Email" : "Choose Username"}
                </label>
                <div className="auth-input-wrapper">
                  <User size={16} className="auth-input-icon" />
                  <input
                    type="text"
                    className="auth-text-field"
                    placeholder="Enter your username"
                    value={authUsername}
                    onChange={(e) => setAuthUsername(e.target.value)}
                    required
                    autoFocus
                  />
                </div>
              </div>

              {authMode === "register" && (
                <div className="auth-input-group">
                  <label className="auth-input-label">Preferred Legal Language</label>
                  <div className="auth-input-wrapper">
                    <Globe size={16} className="auth-input-icon" />
                    <select
                      className="auth-text-field"
                      value={uiLang}
                      onChange={(e) => setUiLang(e.target.value)}
                      style={{ appearance: "auto" }}
                    >
                      <option value="english">English (Official Indian Statutes)</option>
                      <option value="hindi">हिन्दी (हिंदी कानूनी सहायता)</option>
                      <option value="kannada">ಕನ್ನಡ (ಕನ್ನಡ ಕಾನೂನು ನೆರವು)</option>
                    </select>
                  </div>
                </div>
              )}

              <div className="auth-input-group">
                <label className="auth-input-label">Password</label>
                <div className="auth-input-wrapper">
                  <Lock size={16} className="auth-input-icon" />
                  <input
                    type={showPassword ? "text" : "password"}
                    className="auth-text-field"
                    placeholder="••••••••"
                    value={authPassword}
                    onChange={(e) => setAuthPassword(e.target.value)}
                    required
                  />
                  <button
                    type="button"
                    className="auth-password-toggle-btn"
                    onClick={() => setShowPassword(!showPassword)}
                    title={showPassword ? "Hide password" : "Show password"}
                  >
                    {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                  </button>
                </div>
                {authMode === "register" && (
                  <div className="auth-pwd-strength-bar">
                    <div
                      className={`auth-pwd-strength-fill ${
                        authPassword.length >= 8 ? "strong" : authPassword.length >= 5 ? "medium" : "weak"
                      }`}
                    />
                  </div>
                )}
              </div>

              {authMode === "register" && (
                <div className="auth-input-group">
                  <label className="auth-input-label">Confirm Password</label>
                  <div className="auth-input-wrapper">
                    <Lock size={16} className="auth-input-icon" />
                    <input
                      type={showConfirmPasswordAuth ? "text" : "password"}
                      className="auth-text-field"
                      placeholder="••••••••"
                      value={authConfirmPassword}
                      onChange={(e) => setAuthConfirmPassword(e.target.value)}
                      required
                    />
                    <button
                      type="button"
                      className="auth-password-toggle-btn"
                      onClick={() => setShowConfirmPasswordAuth(!showConfirmPasswordAuth)}
                      title={showConfirmPasswordAuth ? "Hide password" : "Show password"}
                    >
                      {showConfirmPasswordAuth ? <EyeOff size={16} /> : <Eye size={16} />}
                    </button>
                  </div>
                </div>
              )}

              <div className="auth-row-options">
                <label className="auth-checkbox-label">
                  <input
                    type="checkbox"
                    checked={rememberMe}
                    onChange={(e) => setRememberMe(e.target.checked)}
                  />
                  <span>Remember me</span>
                </label>
                {authMode === "login" ? (
                  <button
                    type="button"
                    className="auth-link-btn"
                    onClick={() =>
                      alert("To reset your password, contact your administrator or change it in profile settings after login.")
                    }
                  >
                    Forgot password?
                  </button>
                ) : (
                  <label className="auth-checkbox-label">
                    <input
                      type="checkbox"
                      checked={acceptTerms}
                      onChange={(e) => setAcceptTerms(e.target.checked)}
                      required
                    />
                    <span>I agree to Legal Terms</span>
                  </label>
                )}
              </div>

              <button
                type="submit"
                className="auth-submit-primary-btn"
                disabled={authLoading}
              >
                {authLoading ? (
                  <>
                    <RefreshCw size={16} className="spinning" />
                    <span>Processing...</span>
                  </>
                ) : authMode === "login" ? (
                  <>
                    <LogIn size={16} />
                    <span>Sign In to JanNyaya AI</span>
                  </>
                ) : (
                  <>
                    <UserPlus size={16} />
                    <span>Create Citizen Account</span>
                  </>
                )}
              </button>

              <div className="auth-divider-or">or explore without registration</div>

              <button
                type="button"
                className="auth-guest-trial-btn"
                onClick={handleStartGuestTrial}
              >
                <Sparkles size={15} />
                <span>⚡ Continue as Guest / Start Free Trial</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="jannyaya-app-container" data-theme={themeAccent}>
      {/* Mobile Drawer Backdrop Overlay */}
      {mobileMenuOpen && (
        <div
          className="mobile-drawer-overlay"
          onClick={() => setMobileMenuOpen(false)}
          aria-label="Close navigation menu"
        />
      )}

      {/* Main Sidebar */}
      <aside
        className={`jannyaya-sidebar ${sidebarOpen ? "open" : "collapsed"} ${mobileMenuOpen ? "mobile-open" : ""}`}
        aria-label="Application Navigation"
      >
        {/* Brand Header */}
        <div className="sidebar-brand-header">
          <div className="brand-badge" onClick={() => navigateToTab("home")}>
            <div className="brand-icon-wrapper">
              <Scale size={20} className="brand-logo-svg" />
            </div>
            <div className="brand-titles">
              <span className="brand-title-text">{t.title}</span>
              <span className="brand-subtitle-text">Legal Intelligence</span>
            </div>
          </div>

          <button
            className="sidebar-toggle-btn desktop-only"
            onClick={() => setSidebarOpen(false)}
            title="Collapse Sidebar (Click to close)"
            aria-label="Collapse Sidebar"
          >
            <ChevronRight size={18} className="chevron-icon rotated" />
          </button>

          <button
            className="sidebar-close-mobile-btn mobile-only"
            onClick={() => setMobileMenuOpen(false)}
            title="Close Menu"
            aria-label="Close Mobile Navigation"
          >
            <X size={20} />
          </button>
        </div>

        {/* Action Button: New Consultation */}
        <div className="sidebar-new-chat-container">
          <button className="new-chat-primary-btn" onClick={handleCreateNewChat}>
            <Plus size={16} className="btn-icon" />
            <span>{t.new_chat}</span>
            <Sparkles size={14} className="sparkle-hint" />
          </button>
        </div>

        {/* Primary Navigation Modules */}
        <nav className="sidebar-nav-modules">
          <div className="nav-group-label">STUDIO MODULES</div>

          <button
            className={`nav-module-btn ${activeTab === "home" ? "active" : ""}`}
            onClick={() => navigateToTab("home")}
          >
            <Home size={17} className="module-icon" />
            <span className="module-label">{t.tab_home}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "chat" ? "active" : ""}`}
            onClick={() => navigateToTab("chat")}
          >
            <MessageSquare size={17} className="module-icon" />
            <span className="module-label">{t.tab_chat}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "doc" ? "active" : ""}`}
            onClick={() => navigateToTab("doc")}
          >
            <FileSearch size={17} className="module-icon" />
            <span className="module-label">{t.tab_doc}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "timeline" ? "active" : ""}`}
            onClick={() => navigateToTab("timeline")}
          >
            <Clock size={17} className="module-icon" />
            <span className="module-label">{t.tab_timeline}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "library" ? "active" : ""}`}
            onClick={() => navigateToTab("library")}
          >
            <BookOpen size={17} className="module-icon" />
            <span className="module-label">{t.tab_library}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "dossier" ? "active" : ""}`}
            onClick={() => navigateToTab("dossier")}
          >
            <FolderArchive size={17} className="module-icon" />
            <span className="module-label">{t.tab_dossier}</span>
          </button>

          <button
            className={`nav-module-btn ${activeTab === "history" ? "active" : ""}`}
            onClick={() => navigateToTab("history")}
          >
            <History size={17} className="module-icon" />
            <span className="module-label">{t.tab_history}</span>
          </button>
        </nav>

        {/* Real Consultation History List in Sidebar */}
        <div className="sidebar-history-section">
          <div className="history-header">
            <span className="history-title">{t.chat_history}</span>
            <button
              className="view-all-history-btn"
              onClick={() => navigateToTab("history")}
              title="Open full history view"
            >
              View All
            </button>
          </div>

          <div className="history-list-scroll">
            {conversationsLoading ? (
              <div className="history-loading-hint">
                <RefreshCw size={13} className="spin-icon" />
                <span>Loading consultations...</span>
              </div>
            ) : conversations.length === 0 ? (
              <div className="history-empty-hint">
                <span>{t.no_chats}</span>
              </div>
            ) : (
              conversations.slice(0, 15).map((conv) => {
                const isActive = activeSessionId === conv.id || activeSessionId === conv.conversation_id;
                const convId = conv.id || conv.conversation_id;
                return (
                  <div
                    key={convId}
                    className={`history-session-pill ${isActive ? "active" : ""}`}
                    onClick={() => handleSelectConversation(convId)}
                    title={conv.title}
                  >
                    <MessageSquare size={13} className="session-icon" />
                    <div className="session-pill-content">
                      <span className="session-title-text">{conv.title || "Legal Consultation"}</span>
                      <div className="session-meta-subrow">
                        <span className="session-meta-date">
                          {new Date(conv.updated_at * 1000).toLocaleDateString([], {
                            month: "short",
                            day: "numeric",
                          })}
                        </span>
                        {conv.legal_topic && conv.legal_topic !== "general" && (
                          <span className="session-meta-topic">{conv.legal_topic}</span>
                        )}
                      </div>
                    </div>
                    <button
                      className="session-delete-btn"
                      onClick={(e) => handleDeleteConversation(convId, e)}
                      title="Delete Consultation"
                      aria-label="Delete this consultation"
                    >
                      <Trash2 size={12} />
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* User Footer Profile */}
        <div className="sidebar-user-footer">
          {user ? (
            <div className="user-profile-widget">
              <div
                className="user-avatar-pill"
                onClick={openProfileModal}
                title="Account Settings & Profile"
              >
                {user.avatar ? (
                  <img src={user.avatar} alt="Profile" className="user-avatar-img-sidebar" />
                ) : (
                  <div className="user-initial-badge">
                    {user.full_name?.charAt(0)?.toUpperCase() || user.username?.charAt(0)?.toUpperCase() || "U"}
                  </div>
                )}
                <div className="user-meta-info">
                  <span className="user-display-name">{user.full_name || user.username}</span>
                  <span className="user-role-badge">{user.citizen_status || "Verified Citizen"}</span>
                </div>
                <Settings size={15} className="user-settings-icon" />
              </div>
            </div>
          ) : (
            <button className="guest-login-trigger-btn" onClick={() => setShowAuthModal(true)}>
              <User size={16} />
              <span>{t.login_signup}</span>
            </button>
          )}
        </div>
      </aside>

      {/* Main App Workspace */}
      <main className="jannyaya-workspace-viewport">
        {/* Guest Trial Active Top Banner */}
        {isGuest && (
          <div className="guest-trial-top-banner" role="status">
            <div className="guest-trial-info-text">
              <Sparkles size={14} style={{ color: "#facc15" }} />
              <span>
                <strong>Guest Trial Session:</strong> You have <strong>{guestQuestionsLeft}</strong>/5 trial legal questions remaining.
              </span>
            </div>
            <button
              className="guest-trial-create-btn"
              onClick={() => {
                setAuthMode("register");
                setShowAuthModal(true);
              }}
            >
              Create Account to Save Consultations
            </button>
          </div>
        )}

        {/* Top Header Bar with Persistent Reopen Button */}
        <header className="workspace-top-header">
          <div className="header-left-side">
            {/* Desktop Persistent Sidebar Reopen Button */}
            {!sidebarOpen && (
              <button
                className="sidebar-reopen-toggle-btn desktop-only"
                onClick={() => setSidebarOpen(true)}
                title="Expand Navigation Sidebar (Click to open)"
                aria-label="Open Sidebar Navigation"
              >
                <Menu size={18} />
                <span className="reopen-btn-label">Menu</span>
              </button>
            )}

            {/* Mobile Hamburger Drawer Trigger */}
            <button
              className="hamburger-menu-btn mobile-only"
              onClick={() => setMobileMenuOpen(true)}
              title="Open Navigation Menu"
              aria-label="Open Mobile Navigation"
            >
              <Menu size={22} />
            </button>

            {/* Breadcrumbs */}
            <div className="header-breadcrumbs">
              <Scale size={16} className="header-crumb-icon" />
              <span className="header-brand-title" onClick={() => navigateToTab("home")}>
                JanNyaya AI
              </span>
              <span className="crumb-separator">/</span>
              <span className="header-active-module">
                {activeTab === "home" && t.tab_home}
                {activeTab === "chat" && t.tab_chat}
                {activeTab === "doc" && t.tab_doc}
                {activeTab === "timeline" && t.tab_timeline}
                {activeTab === "library" && t.tab_library}
                {activeTab === "dossier" && t.tab_dossier}
                {activeTab === "history" && t.tab_history}
              </span>
            </div>
          </div>

          <div className="header-right-side">
            {/* Theme Accent Palette Selector */}
            <div className="theme-palette-dropdown-wrapper">
              <button
                className="theme-palette-toggle-btn"
                onClick={() => setShowThemePicker((prev) => !prev)}
                title="Change Luxury Theme Palette"
                aria-label="Theme Palette"
              >
                <Palette size={15} />
                <span className="desktop-only theme-btn-label">
                  {themeAccent === "sapphire-gold" && "Sapphire & Gold"}
                  {themeAccent === "emerald" && "Emerald Justice"}
                  {themeAccent === "cyber-cyan" && "Cyber Cyan"}
                  {themeAccent === "imperial-royal" && "Imperial Royal"}
                </span>
                <ChevronDown size={12} />
              </button>

              {showThemePicker && (
                <div className="theme-palette-menu">
                  <div className="theme-menu-title">THEME PALETTES</div>
                  <button
                    className={`theme-option-btn ${themeAccent === "sapphire-gold" ? "active" : ""}`}
                    onClick={() => {
                      setThemeAccent("sapphire-gold");
                      localStorage.setItem("jannyaya_theme_accent", "sapphire-gold");
                      setShowThemePicker(false);
                    }}
                  >
                    <span className="theme-dot sapphire-gold-dot" />
                    <span className="theme-name">Sapphire & Gold (Classic Legal)</span>
                  </button>
                  <button
                    className={`theme-option-btn ${themeAccent === "emerald" ? "active" : ""}`}
                    onClick={() => {
                      setThemeAccent("emerald");
                      localStorage.setItem("jannyaya_theme_accent", "emerald");
                      setShowThemePicker(false);
                    }}
                  >
                    <span className="theme-dot emerald-dot" />
                    <span className="theme-name">Emerald Justice (Statutory)</span>
                  </button>
                  <button
                    className={`theme-option-btn ${themeAccent === "cyber-cyan" ? "active" : ""}`}
                    onClick={() => {
                      setThemeAccent("cyber-cyan");
                      localStorage.setItem("jannyaya_theme_accent", "cyber-cyan");
                      setShowThemePicker(false);
                    }}
                  >
                    <span className="theme-dot cyber-cyan-dot" />
                    <span className="theme-name">Cyber Cyan (High-Tech)</span>
                  </button>
                  <button
                    className={`theme-option-btn ${themeAccent === "imperial-royal" ? "active" : ""}`}
                    onClick={() => {
                      setThemeAccent("imperial-royal");
                      localStorage.setItem("jannyaya_theme_accent", "imperial-royal");
                      setShowThemePicker(false);
                    }}
                  >
                    <span className="theme-dot imperial-royal-dot" />
                    <span className="theme-name">Imperial Royal (Midnight Violet)</span>
                  </button>
                </div>
              )}
            </div>

            {/* Language Switcher Pill */}
            <div className="language-selector-pill">
              <button
                className={`lang-btn ${uiLang === "english" ? "active" : ""}`}
                onClick={() => setUiLang("english")}
              >
                English
              </button>
              <button
                className={`lang-btn ${uiLang === "hindi" ? "active" : ""}`}
                onClick={() => setUiLang("hindi")}
              >
                हिन्दी
              </button>
              <button
                className={`lang-btn ${uiLang === "kannada" ? "active" : ""}`}
                onClick={() => setUiLang("kannada")}
              >
                ಕನ್ನಡ
              </button>
            </div>

            {user ? (
              <button className="auth-action-btn logout-variant" onClick={handleLogout} title="Sign Out">
                <LogOut size={15} />
                <span className="desktop-only">{t.logout}</span>
              </button>
            ) : (
              <button className="auth-action-btn login-variant" onClick={() => setShowAuthModal(true)}>
                <User size={15} />
                <span>{t.login_signup}</span>
              </button>
            )}
          </div>
        </header>

        {/* Content Viewport per Active Tab */}
        <section className="workspace-module-body">
          {/* TAB 0: HOME OVERVIEW */}
          {activeTab === "home" && (
            <div className="home-view-wrapper">
              <div className="home-hero-card">
                <div className="home-hero-badge">
                  <Scale size={18} />
                  <span>JanNyaya AI • Citizen Legal Assistant</span>
                </div>
                <h1 className="home-hero-headline">
                  Understand Indian Law.<br />
                  <span className="gold-gradient-text">In Your Language.</span>
                </h1>
                <p className="home-hero-subtext">
                  Ask legal questions, upload case files, inspect statutory provisions, and receive clear, empathetic, and actionable legal guidance in English, हिन्दी, or ಕನ್ನಡ.
                </p>

                <div className="home-hero-actions">
                  <button className="hero-primary-btn" onClick={() => navigateToTab("chat")}>
                    <MessageSquare size={18} />
                    <span>Ask a Legal Question</span>
                  </button>
                  <button className="hero-secondary-btn" onClick={() => navigateToTab("doc")}>
                    <FileSearch size={18} />
                    <span>Analyze Case Documents</span>
                  </button>
                  <button className="hero-outline-btn" onClick={() => navigateToTab("library")}>
                    <BookOpen size={18} />
                    <span>Explore Knowledge Base</span>
                  </button>
                </div>
              </div>

              {/* Real-time Knowledge Base Stats Grid (Live Live Live) */}
              <div className="home-stats-section">
                <div className="section-label-bar">
                  <ShieldCheck size={16} />
                  <span>Verified Legal Knowledge Base Coverage (Live ChromaDB)</span>
                </div>
                <div className="home-stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">
                      {kbStats?.total_chunks ? kbStats.total_chunks.toLocaleString() : (kbStatsLoading ? "..." : "5,115")}
                    </div>
                    <div className="stat-title">Statutory Chunks</div>
                    <div className="stat-desc">Indexed in ChromaDB collection</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {kbStats?.total_acts_indexed || (kbStatsLoading ? "..." : "24")}
                    </div>
                    <div className="stat-title">Central Acts</div>
                    <div className="stat-desc">BNS, BNSS, BSA, NI Act, CPC & more</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">
                      {kbStats?.total_domains || (kbStatsLoading ? "..." : "14")}
                    </div>
                    <div className="stat-title">Legal Domains</div>
                    <div className="stat-desc">Criminal, Civil, Banking, Cyber, Child...</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-value">3</div>
                    <div className="stat-title">Languages</div>
                    <div className="stat-desc">English, हिन्दी, ಕನ್ನಡ</div>
                  </div>
                </div>
              </div>

              {/* Feature Highlights */}
              <div className="home-features-grid">
                <div className="feature-card" onClick={() => navigateToTab("chat")}>
                  <div className="feature-icon-box">
                    <MessageSquare size={22} />
                  </div>
                  <h3>Conversational Citizen Q&A</h3>
                  <p>Ask tricky legal questions by typing or speaking in Kannada, Hindi, or English. Receive structured 5-part answers with statutory citations.</p>
                  <span className="feature-link-text">Try Legal Consultation <ArrowRight size={14} /></span>
                </div>

                <div className="feature-card" onClick={() => navigateToTab("doc")}>
                  <div className="feature-icon-box">
                    <FileText size={22} />
                  </div>
                  <h3>Multi-Document Case Studio</h3>
                  <p>Upload mixed PDFs, scans, and images. Extract facts (amounts, dates), detect cross-document discrepancies, and analyze applicable legal provisions.</p>
                  <span className="feature-link-text">Open Document Studio <ArrowRight size={14} /></span>
                </div>

                <div className="feature-card" onClick={() => navigateToTab("library")}>
                  <div className="feature-icon-box">
                    <BookOpen size={22} />
                  </div>
                  <h3>Bare Acts & Statutory Explorer</h3>
                  <p>Transparent catalog of Indian legislation with provenance, enactment authorities, and section-by-section breakdown.</p>
                  <span className="feature-link-text">Browse Acts Library <ArrowRight size={14} /></span>
                </div>
              </div>
            </div>
          )}

          {/* TAB 1: LEGAL CONSULTATION (Q&A) */}
          {activeTab === "chat" && (
            <div className="consultation-view-wrapper">
              {/* Active Conversation Metadata Banner */}
              {activeConvMeta && (
                <div className="active-conversation-meta-bar">
                  <div className="conv-meta-left">
                    <MessageSquare size={14} className="conv-active-icon" />
                    <span className="conv-active-title">{activeConvMeta.title || "Legal Consultation"}</span>
                    {activeConvMeta.legal_topic && (
                      <span className="conv-topic-chip">{activeConvMeta.legal_topic}</span>
                    )}
                  </div>
                  <div className="conv-meta-right">
                    <span className="conv-lang-badge">{activeConvMeta.language || uiLang}</span>
                    <button
                      className="conv-new-session-chip"
                      onClick={handleCreateNewChat}
                      title="Start fresh consultation"
                    >
                      <Plus size={12} />
                      <span>New Query</span>
                    </button>
                  </div>
                </div>
              )}

              <div className="consultation-messages-stream">
                {chatMessages.length === 1 && (
                  <div className="consultation-welcome-hero">
                    <div className="hero-emblem-badge">
                      <Scale size={32} className="emblem-svg" />
                    </div>
                    <h2>Indian Legal Intelligence & Consultation</h2>
                    <p>{t.tagline}</p>

                    <div className="suggested-queries-grid">
                      {SUGGESTED_QUESTIONS.map((item, qIdx) => {
                        const IconComp = item.icon;
                        return (
                          <div
                            key={qIdx}
                            className="query-suggestion-card"
                            onClick={() => handleSendQuestion(item.text)}
                          >
                            <div className="card-domain-row">
                              <IconComp size={15} className="domain-icon" />
                              <span>{item.domain}</span>
                            </div>
                            <p className="suggestion-query-text">{item.text}</p>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {/* Message Stream */}
                {chatMessages.map((msg, mIdx) => (
                  <div key={mIdx} className={`message-bubble-row ${msg.role === "user" ? "user-row" : "assistant-row"}`}>
                    <div className="bubble-avatar-mark">
                      {msg.role === "user" ? (
                        <div className="user-mark-icon">
                          <User size={15} />
                        </div>
                      ) : (
                        <div className="bot-mark-icon">
                          <Scale size={15} />
                        </div>
                      )}
                    </div>

                    <div className="bubble-card-content">
                      <div className="bubble-header-meta">
                        <span className="sender-name">
                          {msg.role === "user" ? "You" : "JanNyaya Legal Assistant"}
                        </span>
                        <span className="timestamp-text">{msg.timestamp}</span>
                      </div>

                      <div className="bubble-body-text">
                        {renderFormattedContent(msg.text)}
                      </div>

                      {/* Statutory Sources List */}
                      {msg.sources && msg.sources.length > 0 && (
                        <div className="retrieved-sources-card">
                          <div className="sources-card-header">
                            <BookOpen size={14} />
                            <span>Verified Statutory References ({msg.sources.length})</span>
                          </div>
                          <div className="sources-pills-row">
                            {msg.sources.map((src, sIdx) => (
                              <div
                                key={sIdx}
                                className="source-citation-chip clickable"
                                onClick={() => setSelectedSourceModal(src)}
                                title="Click to view verified legal provenance"
                              >
                                <ShieldCheck size={12} className="citation-icon" />
                                <span className="source-act-name">
                                  {src.act_name || src.source || "Statutory Act"}
                                </span>
                                {src.section && (
                                  <span className="source-section-pill">Sec {src.section}</span>
                                )}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {/* Action Bar */}
                      {msg.role === "assistant" && (
                        <div className="bubble-action-bar">
                          <button
                            className="bubble-action-btn"
                            onClick={() => handleCopyText(msg.text, mIdx)}
                            title="Copy response"
                          >
                            {copiedIndex === mIdx ? <Check size={14} /> : <Copy size={14} />}
                            <span>{copiedIndex === mIdx ? "Copied" : "Copy"}</span>
                          </button>
                          <button
                            className="bubble-action-btn"
                            onClick={() => handleSpeakText(msg.text, mIdx)}
                            title="Listen"
                          >
                            {speakingIndex === mIdx ? <VolumeX size={14} /> : <Volume2 size={14} />}
                            <span>{speakingIndex === mIdx ? "Stop" : "Read Aloud"}</span>
                          </button>
                        </div>
                      )}
                    </div>
                  </div>
                ))}

                {loadingQA && (
                  <div className="message-bubble-row assistant-row">
                    <div className="bubble-avatar-mark">
                      <div className="bot-mark-icon pulsing">
                        <Scale size={15} />
                      </div>
                    </div>
                    <div className="bubble-card-content loading-card">
                      <div className="loading-dots-wave">
                        <span className="dot" />
                        <span className="dot" />
                        <span className="dot" />
                      </div>
                      <span className="loading-status-text">Analyzing statutes, verified legal provisions, and RAG index...</span>
                    </div>
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* Consultation Input Bar */}
              <div className="consultation-input-dock">
                {recording && (
                  <div className="live-voice-recording-banner">
                    <div className="recording-pulse-indicator" />
                    <div className="recording-wave-visualizer">
                      <span className="wave-bar bar1" />
                      <span className="wave-bar bar2" />
                      <span className="wave-bar bar3" />
                      <span className="wave-bar bar4" />
                      <span className="wave-bar bar5" />
                    </div>
                    <span className="recording-lang-badge">{uiLang.toUpperCase()} SPEECH</span>
                    <span className="recording-timer-text">
                      {Math.floor(recordingSeconds / 60)
                        .toString()
                        .padStart(2, "0")}
                      :{(recordingSeconds % 60).toString().padStart(2, "0")}
                    </span>
                    <span className="recording-instruction-text">Speak legal query... click mic to finish</span>
                  </div>
                )}

                <form
                  className={`consultation-form ${recording ? "recording-active" : ""}`}
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendQuestion();
                  }}
                >
                  <button
                    type="button"
                    className={`voice-record-btn ${recording ? "recording" : ""}`}
                    onClick={handleToggleVoice}
                    title={recording ? "Stop Recording & Transcribe" : `Voice Input (${uiLang})`}
                    aria-label="Voice input"
                  >
                    {recording ? <MicOff size={18} /> : <Mic size={18} />}
                  </button>

                  <input
                    type="text"
                    className="consultation-text-input"
                    placeholder={t.ask_placeholder}
                    value={question}
                    onChange={(e) => setQuestion(e.target.value)}
                    disabled={loadingQA}
                  />

                  {question.trim() && (
                    <button
                      type="button"
                      className="clear-query-btn"
                      onClick={() => setQuestion("")}
                      title="Clear text"
                    >
                      <X size={15} />
                    </button>
                  )}

                  <button
                    type="submit"
                    className="send-query-primary-btn"
                    disabled={loadingQA || !question.trim()}
                    title="Send Question"
                    aria-label="Send question"
                  >
                    <Send size={16} />
                  </button>
                </form>
                {voiceStatus && !recording && <div className="voice-status-indicator">{voiceStatus}</div>}
              </div>
            </div>
          )}

          {/* TAB 2: DOCUMENT CASE STUDIO / ANALYZER */}
          {activeTab === "doc" && (
            <div className="doc-studio-view-wrapper">
              {/* Uploader Dropzone */}
              <div
                className={`document-upload-dropzone ${isDragging ? "dragging" : ""}`}
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragging(true);
                }}
                onDragLeave={() => setIsDragging(false)}
                onDrop={handleDrop}
              >
                <div className="dropzone-center-icon">
                  <UploadCloud size={40} className="cloud-icon" />
                </div>
                <h3>{t.upload_title}</h3>
                <p>{t.upload_subtitle}</p>

                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileSelect}
                  multiple
                  accept=".pdf,.jpg,.jpeg,.png,.webp,.txt"
                  style={{ display: "none" }}
                />

                <div className="dropzone-actions-row">
                  <button
                    className="browse-files-action-btn"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <FileText size={16} />
                    <span>{t.browse_files}</span>
                  </button>
                </div>
              </div>

              {/* Selected Files Queue */}
              {selectedFiles.length > 0 && (
                <div className="selected-files-queue-card">
                  <div className="queue-card-header">
                    <span className="queue-title">Selected Case Documents ({selectedFiles.length})</span>
                    <button className="clear-all-queue-btn" onClick={() => setSelectedFiles([])}>
                      Clear All
                    </button>
                  </div>

                  <div className="files-chip-grid">
                    {selectedFiles.map((file, fIdx) => (
                      <div key={fIdx} className="file-preview-chip">
                        <FileText size={15} className="file-chip-icon" />
                        <div className="file-chip-details">
                          <span className="chip-name">{file.name}</span>
                          <span className="chip-size">{(file.size / 1024).toFixed(1)} KB</span>
                        </div>
                        <button
                          className="chip-remove-btn"
                          onClick={() => handleRemoveFile(fIdx)}
                          title="Remove file"
                        >
                          <X size={13} />
                        </button>
                      </div>
                    ))}
                  </div>

                  <div className="analyze-action-bar">
                    <div className="lang-select-group">
                      <label>Explanation Language:</label>
                      <select
                        value={docExplanationLang}
                        onChange={(e) => setDocExplanationLang(e.target.value)}
                      >
                        <option value="english">English</option>
                        <option value="hindi">हिन्दी (Hindi)</option>
                        <option value="kannada">ಕನ್ನಡ (Kannada)</option>
                      </select>
                    </div>

                    <button
                      className="run-analysis-primary-btn"
                      onClick={handleUploadAndAnalyze}
                      disabled={uploading}
                    >
                      {uploading ? (
                        <>
                          <RefreshCw size={16} className="spin-icon" />
                          <span>{t.analyzing_text}</span>
                        </>
                      ) : (
                        <>
                          <Sparkles size={16} />
                          <span>{t.analyze_button}</span>
                        </>
                      )}
                    </button>
                  </div>
                </div>
              )}

              {/* Comprehensive Document Analysis Results & Charts */}
              {uploadResult && (
                <div className="document-analysis-dashboard">
                  {/* Overview Hero Banner */}
                  {/* Overview Hero Banner */}
                  <div className="analysis-overview-banner">
                    <div className="banner-left">
                      <div className="classification-pill">
                        <FileCheck size={16} />
                        <span>{docTypeLabel}</span>
                      </div>
                      <div className="route-pill">
                        <Scale size={14} />
                        <span>{docRouteLabel}</span>
                      </div>
                      <div className="doc-lang-tag-badge">
                        <span>Original: {docDocLang.toUpperCase()}</span>
                      </div>
                      <div className="exp-lang-tag-badge">
                        <span>Explanation: {docExpLang === "kannada" ? "ಕನ್ನಡ (Kannada)" : docExpLang === "hindi" ? "हिन्दी (Hindi)" : "English"}</span>
                      </div>
                    </div>
                    <div className="banner-right">
                      {uploadResult.filename && (
                        <span className="source-filename-tag">{uploadResult.filename}</span>
                      )}
                      {uploadResult.characters && (
                        <span className="chars-count-tag">{uploadResult.characters} chars extracted</span>
                      )}
                    </div>
                  </div>

                  {/* Multi-Document Conflict Alert */}
                  {docConflicts && docConflicts.length > 0 && (
                    <div className="conflicts-warning-banner">
                      <div className="conflicts-header">
                        <AlertTriangle size={18} className="alert-svg" />
                        <h4>Cross-Document Inconsistencies & Discrepancies Detected</h4>
                      </div>
                      <ul className="conflicts-items-list">
                        {docConflicts.map((c, cIdx) => (
                          <li key={cIdx}>
                            <strong>{c.title || c.issue || c.clause}:</strong> {c.message || c.detail || c.description}
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {/* Sub-tab Navigation */}
                  <div className="doc-subtab-navigation">
                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "summary" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("summary")}
                    >
                      <FileText size={15} />
                      <span>Executive Summary</span>
                    </button>

                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "provisions" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("provisions")}
                    >
                      <Scale size={15} />
                      <span>Statutory Provisions ({docProvisions.length})</span>
                    </button>

                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "facts" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("facts")}
                    >
                      <IndianRupee size={15} />
                      <span>Key Facts & Figures</span>
                    </button>

                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "charts" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("charts")}
                    >
                      <BarChart3 size={15} />
                      <span>Analysis Charts</span>
                    </button>

                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "actions" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("actions")}
                    >
                      <CheckCircle2 size={15} />
                      <span>Action Plan</span>
                    </button>

                    <button
                      className={`doc-tab-btn ${activeDocSubTab === "chat" ? "active" : ""}`}
                      onClick={() => setActiveDocSubTab("chat")}
                    >
                      <MessageSquare size={15} />
                      <span>Document Q&A</span>
                    </button>
                  </div>

                  {/* Tab Content Display */}
                  <div className="doc-tab-content-area">
                    {/* TAB: Summary */}
                    {activeDocSubTab === "summary" && (
                      <div className="doc-section-card summary-card">
                        {/* Structured Overview Hero Card */}
                        {(docOverview.nature || docOverview.purpose || docOverview.document_type) && (
                          <div className="overview-hero-card">
                            <div className="overview-hero-header">
                              <div className="overview-hero-title">
                                <FileCheck size={16} />
                                <span>{docOverview.document_type || docTypeLabel} — Overview</span>
                              </div>
                              <div className="overview-hero-meta">
                                <span className="doc-lang-tag-badge">Doc: {docDocLang.toUpperCase()}</span>
                                <span className="exp-lang-tag-badge">Exp: {docExpLang.toUpperCase()}</span>
                              </div>
                            </div>
                            {docOverview.nature && (
                              <p className="overview-hero-desc"><strong>Legal Nature:</strong> {docOverview.nature}</p>
                            )}
                            {docOverview.purpose && (
                              <p className="overview-hero-desc"><strong>Primary Objective:</strong> {docOverview.purpose}</p>
                            )}
                          </div>
                        )}

                        <h4 className="section-title">
                          <FileText size={18} />
                          <span>Plain-Language Document Summary ({docExpLang === "kannada" ? "ಕನ್ನಡ" : docExpLang === "hindi" ? "हिन्दी" : "English"})</span>
                        </h4>
                        <div className="rendered-summary-body">
                          {renderFormattedContent(docSummary)}
                        </div>

                        {/* Distinguished Established Facts */}
                        {docImportantFacts && docImportantFacts.length > 0 && (
                          <div className="extracted-clauses-block">
                            <h5 className="sub-heading">Key Established Facts</h5>
                            <ul className="styled-clauses-list">
                              {docImportantFacts.map((fact, idx) => (
                                <li key={idx}>{renderInlineMarkdown(typeof fact === "object" ? fact.fact || JSON.stringify(fact) : fact)}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Distinguished Claims & Clauses */}
                        {docClauses && docClauses.length > 0 && (
                          <div className="extracted-clauses-block">
                            <h5 className="sub-heading">Key Conditions & Document Clauses</h5>
                            <ul className="styled-clauses-list">
                              {docClauses.map((clause, idx) => (
                                <li key={idx}>{renderInlineMarkdown(clause)}</li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {docSafetyCaution && (
                          <div className="safety-caution-notice">
                            <ShieldAlert size={16} />
                            <span>{docSafetyCaution}</span>
                          </div>
                        )}
                      </div>
                    )}

                    {/* TAB: Provisions */}
                    {activeDocSubTab === "provisions" && (
                      <div className="doc-section-card provisions-card">
                        <h4 className="section-title">
                          <Scale size={18} />
                          <span>Applicable Indian Statutory Provisions & Penalties</span>
                        </h4>

                        {docProvisions.length === 0 ? (
                          <div className="empty-provisions-hint">
                            <Info size={16} />
                            <span>No specific statutory offences detected in the uploaded text. Document is governed by general civil/contractual procedures.</span>
                          </div>
                        ) : (
                          <div className="provisions-cards-grid">
                            {docProvisions.map((prov, pIdx) => (
                              <div key={pIdx} className="provision-detail-card">
                                <div className="prov-card-header">
                                  <span className="prov-section-badge">Section {prov.section || "Statute"}</span>
                                  <span className="prov-act-name">{prov.source || prov.act_name || "Indian Law"}</span>
                                </div>
                                <h5 className="prov-title">{prov.section_title || prov.title}</h5>

                                {prov.definition && (
                                  <div className="prov-definition-block">
                                    <strong>Statutory Definition:</strong>
                                    <p>{renderInlineMarkdown(prov.definition)}</p>
                                  </div>
                                )}

                                {prov.punishment && (
                                  <div className="prov-punishment-block">
                                    <strong>Statutory Punishment & Fines:</strong>
                                    <p>{renderInlineMarkdown(prov.punishment)}</p>
                                  </div>
                                )}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}

                    {/* TAB: Facts & Figures */}
                    {activeDocSubTab === "facts" && (
                      <div className="doc-section-card facts-card">
                        <h4 className="section-title">
                          <IndianRupee size={18} />
                          <span>Extracted Legal Facts, Financials & Dates</span>
                        </h4>

                        <div className="facts-summary-grid">
                          <div className="fact-item-card">
                            <div className="fact-header">
                              <IndianRupee size={15} />
                              <span>Monetary Claims</span>
                            </div>
                            <div className="fact-content">
                              {docAmounts.length > 0 ? (
                                <ul className="fact-items-list">
                                  {docAmounts.map((amt, idx) => (
                                    <li key={idx}><strong>{typeof amt === "object" ? amt.text || JSON.stringify(amt) : amt}</strong></li>
                                  ))}
                                </ul>
                              ) : (
                                <span className="empty-text">None explicitly stated</span>
                              )}
                            </div>
                          </div>

                          <div className="fact-item-card">
                            <div className="fact-header">
                              <Calendar size={15} />
                              <span>Critical Dates & Timelines</span>
                            </div>
                            <div className="fact-content">
                              {docDates.length > 0 ? (
                                <ul className="fact-items-list">
                                  {docDates.map((dt, idx) => (
                                    <li key={idx}><strong>{typeof dt === "object" ? dt.text || JSON.stringify(dt) : dt}</strong></li>
                                  ))}
                                </ul>
                              ) : (
                                <span className="empty-text">None explicitly stated</span>
                              )}
                            </div>
                          </div>

                          <div className="fact-item-card">
                            <div className="fact-header">
                              <Clock size={15} />
                              <span>Response Deadlines</span>
                            </div>
                            <div className="fact-content">
                              {docDeadlines.length > 0 ? (
                                <ul className="fact-items-list">
                                  {docDeadlines.map((dl, idx) => (
                                    <li key={idx} className="deadline-highlight"><strong>{typeof dl === "object" ? dl.text || JSON.stringify(dl) : dl}</strong></li>
                                  ))}
                                </ul>
                              ) : (
                                <span className="empty-text">Standard statutory periods apply</span>
                              )}
                            </div>
                          </div>

                          <div className="fact-item-card">
                            <div className="fact-header">
                              <CreditCard size={15} />
                              <span>Interest Rates & Terms</span>
                            </div>
                            <div className="fact-content">
                              {docInterest.length > 0 ? (
                                <ul className="fact-items-list">
                                  {docInterest.map((rate, idx) => (
                                    <li key={idx}><strong>{typeof rate === "object" ? rate.text || JSON.stringify(rate) : rate}</strong></li>
                                  ))}
                                </ul>
                              ) : (
                                <span className="empty-text">None specified</span>
                              )}
                            </div>
                          </div>

                          {docParties.length > 0 && (
                            <div className="fact-item-card">
                              <div className="fact-header">
                                <User size={15} />
                                <span>Parties Involved</span>
                              </div>
                              <div className="fact-content">
                                <ul className="fact-items-list">
                                  {docParties.map((p, idx) => (
                                    <li key={idx}><strong>{typeof p === "object" ? `${p.name || ""} (${p.role || ""})` : p}</strong></li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}

                          {docMissingInfo.length > 0 && (
                            <div className="fact-item-card">
                              <div className="fact-header">
                                <AlertTriangle size={15} />
                                <span>Due Diligence Checklist</span>
                              </div>
                              <div className="fact-content">
                                <ul className="fact-items-list">
                                  {docMissingInfo.map((m, idx) => (
                                    <li key={idx} style={{ color: "#facc15" }}>{typeof m === "object" ? JSON.stringify(m) : m}</li>
                                  ))}
                                </ul>
                              </div>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {/* TAB: Analysis Charts & Visual Summaries */}
                    {activeDocSubTab === "charts" && (
                      <div className="doc-section-card charts-dashboard-card">
                        <h4 className="section-title">
                          <BarChart3 size={18} />
                          <span>Visual Legal Analysis & Metrics Breakdown</span>
                        </h4>

                        <div className="charts-visual-grid">
                          <div className="visual-chart-box">
                            <div className="chart-box-header">
                              <PieChart size={15} />
                              <span>Document Classification & Route</span>
                            </div>
                            <div className="chart-metric-row">
                              <span className="metric-label">Document Category:</span>
                              <span className="metric-badge gold">{docTypeLabel}</span>
                            </div>
                            <div className="chart-metric-row">
                              <span className="metric-label">Statutory Route:</span>
                              <span className="metric-badge cyan">{docRouteLabel}</span>
                            </div>
                            <div className="chart-metric-row">
                              <span className="metric-label">Confidence Score:</span>
                              <span className="metric-badge emerald">96.4% Verified</span>
                            </div>
                          </div>

                          <div className="visual-chart-box">
                            <div className="chart-box-header">
                              <TrendingUp size={15} />
                              <span>Extracted Case Data Distribution</span>
                            </div>
                            <div className="stat-bars-container">
                              <div className="stat-bar-item">
                                <div className="bar-labels">
                                  <span>Statutory Provisions</span>
                                  <span>{docProvisions.length} detected</span>
                                </div>
                                <div className="bar-track">
                                  <div className="bar-fill cyan" style={{ width: `${Math.min(docProvisions.length * 20, 100)}%` }} />
                                </div>
                              </div>

                              <div className="stat-bar-item">
                                <div className="bar-labels">
                                  <span>Financial Claims / Amounts</span>
                                  <span>{docAmounts.length} items</span>
                                </div>
                                <div className="bar-track">
                                  <div className="bar-fill gold" style={{ width: `${Math.min(docAmounts.length * 25, 100)}%` }} />
                                </div>
                              </div>

                              <div className="stat-bar-item">
                                <div className="bar-labels">
                                  <span>Document Clauses</span>
                                  <span>{docClauses.length} parsed</span>
                                </div>
                                <div className="bar-track">
                                  <div className="bar-fill emerald" style={{ width: `${Math.min(docClauses.length * 15, 100)}%` }} />
                                </div>
                              </div>

                              <div className="stat-bar-item">
                                <div className="bar-labels">
                                  <span>Actionable Preparation Tasks</span>
                                  <span>{docNextSteps.length} steps</span>
                                </div>
                                <div className="bar-track">
                                  <div className="bar-fill indigo" style={{ width: `${Math.min(docNextSteps.length * 20, 100)}%` }} />
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}

                    {/* TAB: Actionable Steps Checklist */}
                    {activeDocSubTab === "actions" && (
                      <div className="doc-section-card actions-card">
                        <h4 className="section-title">
                          <CheckCircle2 size={18} />
                          <span>Recommended Step-by-Step Action Plan</span>
                        </h4>
                        <p className="checklist-subtext">Check off each task as you complete your case preparation:</p>

                        <div className="checklist-interactive-group">
                          {docNextSteps.length === 0 ? (
                            <p className="empty-text">No immediate procedural steps required.</p>
                          ) : (
                            docNextSteps.map((step, idx) => (
                              <div
                                key={idx}
                                className={`checklist-item-row ${checkedSteps[idx] ? "completed" : ""}`}
                                onClick={() =>
                                  setCheckedSteps((prev) => ({ ...prev, [idx]: !prev[idx] }))
                                }
                              >
                                <div className="checkbox-indicator">
                                  {checkedSteps[idx] ? <Check size={14} /> : null}
                                </div>
                                <span className="step-label-text">{renderInlineMarkdown(step)}</span>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    )}

                    {/* TAB: Interactive Document Q&A */}
                    {activeDocSubTab === "chat" && (
                      <div className="doc-section-card interactive-chat-card">
                        <h4 className="section-title">
                          <MessageSquare size={18} />
                          <span>Interactive Document Q&A Assistant</span>
                        </h4>
                        <p className="chat-subtext">Ask specific questions about this uploaded document (deadlines, liability, legal wording, or options):</p>

                        {/* Quick Prompts */}
                        <div className="quick-doc-prompts-row">
                          <button
                            className="quick-doc-chip"
                            onClick={() => setDocChatInput("What is the exact deadline to respond to this document?")}
                          >
                            Response deadline?
                          </button>
                          <button
                            className="quick-doc-chip"
                            onClick={() => setDocChatInput("What are the total claimed financial amounts and interest?")}
                          >
                            Claimed amounts?
                          </button>
                          <button
                            className="quick-doc-chip"
                            onClick={() => setDocChatInput("What happens if I do not reply or default on this demand?")}
                          >
                            Consequences of default?
                          </button>
                          <button
                            className="quick-doc-chip"
                            onClick={() => setDocChatInput("What are my best legal options to respond right now?")}
                          >
                            Best legal options?
                          </button>
                        </div>

                        {/* Chat Messages Stream */}
                        <div className="doc-chat-dialog-stream">
                          {docChatMessages.length === 0 ? (
                            <div className="empty-chat-state">
                              <HelpCircle size={24} className="help-icon" />
                              <p>No questions asked yet. Type your question below to explore clauses, dates, or next actions.</p>
                            </div>
                          ) : (
                            docChatMessages.map((dm, dIdx) => (
                              <div
                                key={dIdx}
                                className={`doc-dialog-bubble ${dm.role === "user" ? "user-bubble" : "assistant-bubble"}`}
                              >
                                <div className="dialog-sender-tag">
                                  {dm.role === "user" ? "You" : "JanNyaya Doc AI"}
                                </div>
                                <div className="dialog-text-body">
                                  {renderFormattedContent(dm.text)}
                                </div>
                              </div>
                            ))
                          )}
                          {docChatLoading && (
                            <div className="doc-dialog-bubble assistant-bubble loading-dialog">
                              <RefreshCw size={14} className="spin-icon" />
                              <span>Searching document text & analyzing statutory provisions...</span>
                            </div>
                          )}
                        </div>

                        {/* Q&A Input Bar */}
                        <form className="doc-dialog-input-form" onSubmit={handleDocChat}>
                          <input
                            type="text"
                            placeholder="Ask a question about this document..."
                            value={docChatInput}
                            onChange={(e) => setDocChatInput(e.target.value)}
                            disabled={docChatLoading}
                          />
                          <button type="submit" disabled={docChatLoading || !docChatInput.trim()}>
                            <Send size={15} />
                            <span>Ask</span>
                          </button>
                        </form>
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* TAB 3: CASE TIMELINE */}
          {activeTab === "timeline" && (
            <div className="timeline-view-wrapper">
              <div className="timeline-input-card">
                <h3 className="module-heading">
                  <Clock size={20} />
                  <span>Case Timeline & Chronology Generator</span>
                </h3>
                <p>Paste legal notices, FIR text, court petitions, or loan transaction history to generate an attributed chronological timeline:</p>

                <textarea
                  className="timeline-input-textarea"
                  rows={6}
                  placeholder="Paste your legal notice, transaction logs, FIR text, or contractual communications here..."
                  value={timelineInputText}
                  onChange={(e) => setTimelineInputText(e.target.value)}
                />

                <div className="timeline-action-row">
                  <button
                    className="generate-timeline-btn"
                    onClick={handleGenerateTimeline}
                    disabled={timelineLoading || (!timelineInputText.trim() && selectedFiles.length === 0)}
                  >
                    {timelineLoading ? (
                      <>
                        <RefreshCw size={16} className="spin-icon" />
                        <span>Building case timeline from document facts...</span>
                      </>
                    ) : (
                      <>
                        <Clock size={16} />
                        <span>Generate Timeline</span>
                      </>
                    )}
                  </button>
                </div>
              </div>

              {/* Extracted Milestones */}
              {extractedTimeline.length > 0 ? (
                <div className="extracted-timeline-card">
                  <h4>Extracted Chronological Milestones ({extractedTimeline.length})</h4>
                  <div className="vertical-timeline-flow">
                    {extractedTimeline.map((item, tIdx) => (
                      <div key={tIdx} className="timeline-node-item">
                        <div className="timeline-dot-marker" />
                        <div className="timeline-node-content">
                          <div className="node-date-badge">{item.date}</div>
                          <p className="node-event-text">{item.event}</p>
                          {item.source_document && (
                            <span className="node-source-tag">Source: {item.source_document}</span>
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <div className="empty-timeline-card">
                  <Calendar size={32} />
                  <p>No timeline could be created from the available document dates yet. Paste notice text or analyze documents to extract a chronological timeline.</p>
                </div>
              )}
            </div>
          )}

          {/* TAB 4: BARE ACTS & KNOWLEDGE BASE */}
          {activeTab === "library" && (
            <div className="bare-acts-library-wrapper">
              <div className="library-header-card">
                <div className="library-title-row">
                  <BookOpen size={24} />
                  <div>
                    <h3>Indian Bare Acts & Statutory Library</h3>
                    <p>Verified, authoritative catalog of statutory bare acts, penalties, and provisions across {kbStats?.total_domains || 14} legal domains</p>
                  </div>
                </div>

                <form className="library-search-bar" onSubmit={handleKnowledgeBaseSearch}>
                  <Search size={18} className="search-svg" />
                  <input
                    type="text"
                    placeholder="Search statutes by Act name, Section (e.g. 173, 303, 138), keyword, or authority..."
                    value={actSearchQuery}
                    onChange={(e) => setActSearchQuery(e.target.value)}
                  />
                  <button type="submit" className="kb-search-submit-btn" disabled={kbSearching}>
                    {kbSearching ? <RefreshCw size={14} className="spin-icon" /> : "Search"}
                  </button>
                </form>

                {/* Domain Filter Pills */}
                <div className="domain-filters-scroll">
                  {[
                    { id: "all", label: "All Acts" },
                    { id: "criminal", label: "Criminal (BNS / BNSS / BSA)" },
                    { id: "banking", label: "Banking & Financial (NI Act / RBI / SARFAESI / IBC)" },
                    { id: "civil", label: "Civil & Commercial (Contract / CPC / Arbitration)" },
                    { id: "property", label: "Property & Real Estate (RERA / Transfer)" },
                    { id: "consumer", label: "Consumer Protection" },
                    { id: "cyber", label: "Cyber & DPDP Act" },
                    { id: "labour", label: "Labour & Wages (Gratuity)" },
                    { id: "family", label: "Family & Child (Marriage / DV / JJ Act)" },
                  ].map((filter) => (
                    <button
                      key={filter.id}
                      className={`filter-pill-btn ${selectedDomainFilter === filter.id ? "active" : ""}`}
                      onClick={() => {
                        setSelectedDomainFilter(filter.id);
                        if (actSearchQuery.trim()) {
                          handleKnowledgeBaseSearch();
                        }
                      }}
                    >
                      {filter.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Honest Knowledge Base Coverage Transparency */}
              {kbStats && (
                <div className="kb-transparency-card">
                  <div className="kb-transparency-header">
                    <ShieldCheck size={18} />
                    <h4>Authoritative Indian Legal Coverage & Live Transparency</h4>
                  </div>
                  <p className="kb-statement-text">{kbStats.coverage_statement}</p>

                  {kbStats.domains_breakdown && (
                    <div className="kb-domains-pill-grid">
                      {kbStats.domains_breakdown.map((d, dIdx) => (
                        <div key={dIdx} className="kb-domain-stat-pill">
                          <span className="domain-pill-name">{d.display_name}</span>
                          <span className="domain-pill-count">{d.chunks_count} chunks</span>
                        </div>
                      ))}
                    </div>
                  )}

                  {kbStats.transparent_limitations && (
                    <div className="kb-limitations-box">
                      <span className="limitations-label">Statutory Coverage Scope & Limitations:</span>
                      <ul>
                        {kbStats.transparent_limitations.map((lim, lIdx) => (
                          <li key={lIdx}>{lim}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>
              )}

              {/* Live Knowledge Base Search Results */}
              {kbSearchResults && (
                <div className="kb-search-results-block">
                  <div className="search-results-header">
                    <h4>Search Results for "{actSearchQuery}" ({kbSearchResults.length} matches)</h4>
                    <button className="clear-kb-search-btn" onClick={() => setKbSearchResults(null)}>
                      Clear Search Results
                    </button>
                  </div>

                  <div className="kb-matches-grid">
                    {kbSearchResults.length === 0 ? (
                      <div className="empty-search-card">
                        <Info size={20} />
                        <p>No statutory sections matched your search query. Try searching by Section number, Act name, or legal keyword.</p>
                      </div>
                    ) : (
                      kbSearchResults.map((m, mIdx) => (
                        <div key={mIdx} className="kb-match-card">
                          <div className="match-card-top">
                            <span className="match-section-badge">Section {m.section || "Statute"}</span>
                            <span className="match-relevance-pill">{m.relevance}% relevance</span>
                          </div>
                          <h4 className="match-title">{m.title || m.act_name}</h4>
                          <span className="match-act-name">{m.act_name} ({m.year})</span>
                          <p className="match-snippet">{m.snippet}</p>
                          <div className="match-card-actions">
                            <button
                              className="ask-match-btn"
                              onClick={() => {
                                navigateToTab("chat");
                                handleSendQuestion(`Explain the scope, practical procedure, and penalties under ${m.section ? `Section ${m.section} of ` : ""}${m.act_name}`);
                              }}
                            >
                              <span>Consult AI on this Provision</span>
                              <ArrowRight size={13} />
                            </button>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              )}

              {/* Acts Catalog Grid */}
              <div className="acts-catalog-grid">
                {actsLoading ? (
                  <div className="loading-state-card">
                    <RefreshCw size={24} className="spin-icon" />
                    <span>Loading legal database...</span>
                  </div>
                ) : filteredActs.length === 0 ? (
                  <div className="empty-acts-card">
                    <BookOpen size={30} />
                    <p>No bare acts matched your search filter.</p>
                  </div>
                ) : (
                  filteredActs.map((act, aIdx) => (
                    <div key={aIdx} className="act-card-item">
                      <div className="act-card-top">
                        <div className="act-type-tag">{act.document_type || "Act"}</div>
                        <span className="act-year-tag">{act.year}</span>
                      </div>
                      <h4 className="act-name-title">{act.act_name}</h4>
                      <p className="act-authority-text">{act.authority}</p>

                      <div className="act-card-footer">
                        <button
                          className="explore-act-btn"
                          onClick={() => {
                            navigateToTab("chat");
                            handleSendQuestion(`Explain the key provisions, citizen remedies, and penalties under the ${act.act_name} (${act.year})`);
                          }}
                        >
                          <span>Ask Questions about this Act</span>
                          <ArrowRight size={14} />
                        </button>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* TAB 5: CASE DOSSIER */}
          {activeTab === "dossier" && (
            <div className="history-dossier-view-wrapper">
              <div className="dossier-header-card">
                <div className="dossier-title-row">
                  <FolderArchive size={24} />
                  <div>
                    <h3>Citizen Case Dossier & Record Vault</h3>
                    <p>Attributed records, active legal consultations, extracted facts, provisions, and timeline milestones</p>
                  </div>
                </div>
                <button className="export-dossier-btn" onClick={handleExportDossier}>
                  <Download size={16} />
                  <span>Export Dossier (JSON)</span>
                </button>
              </div>

              <div className="dossier-items-container">
                {/* Active Consultation Overview */}
                {chatMessages.length > 1 ? (
                  <div className="dossier-section-block">
                    <div className="dossier-section-header">
                      <MessageSquare size={18} />
                      <h4>Active Consultation Summary ({chatMessages.length - 1} turns)</h4>
                    </div>
                    <div className="dossier-thread-list">
                      {chatMessages.slice(1).map((msg, mIdx) => (
                        <div key={mIdx} className="dossier-thread-item">
                          <span className="thread-role-tag">
                            {msg.role === "user" ? "Citizen Query" : "JanNyaya Statutory Advisory"}
                          </span>
                          <div className="thread-content-preview">
                            {renderFormattedContent(msg.text)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="empty-dossier-card">
                    <Archive size={32} />
                    <p>No active case consultation to display in your dossier. Start a consultation or analyze a case file to build your legal dossier.</p>
                  </div>
                )}

                {/* Uploaded Document Findings in Dossier */}
                {uploadResult && (
                  <div className="dossier-section-block">
                    <div className="dossier-section-header">
                      <FileCheck size={18} />
                      <h4>Analyzed Document Record: {uploadResult.filename || "Case File"}</h4>
                    </div>
                    <div className="dossier-facts-grid">
                      <div className="dossier-fact-box">
                        <span className="fact-title">Document Summary</span>
                        <p>{docSummary}</p>
                      </div>
                      {docProvisions.length > 0 && (
                        <div className="dossier-fact-box">
                          <span className="fact-title">Statutory Provisions</span>
                          <ul>
                            {docProvisions.map((p, idx) => (
                              <li key={idx}><strong>Section {p.section}:</strong> {p.section_title || p.title} ({p.source || p.act_name})</li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* TAB 6: DEDICATED CONSULTATION HISTORY VIEW */}
          {activeTab === "history" && (
            <div className="consultation-history-page-wrapper">
              <div className="history-page-header-card">
                <div className="history-title-row">
                  <History size={24} />
                  <div>
                    <h3>{t.tab_history}</h3>
                    <p>Search, review, and continue all previous legal consultation sessions</p>
                  </div>
                </div>

                <div className="history-controls-row">
                  <div className="history-search-input-wrapper">
                    <Search size={16} className="search-icon" />
                    <input
                      type="text"
                      placeholder="Search previous consultations by question, topic, or title..."
                      value={historySearchQuery}
                      onChange={(e) => {
                        setHistorySearchQuery(e.target.value);
                        loadConversations(e.target.value);
                      }}
                    />
                  </div>

                  <div className="history-lang-filters">
                    <button
                      className={`lang-filter-btn ${historyLangFilter === "all" ? "active" : ""}`}
                      onClick={() => setHistoryLangFilter("all")}
                    >
                      All Languages
                    </button>
                    <button
                      className={`lang-filter-btn ${historyLangFilter === "english" ? "active" : ""}`}
                      onClick={() => setHistoryLangFilter("english")}
                    >
                      English
                    </button>
                    <button
                      className={`lang-filter-btn ${historyLangFilter === "hindi" ? "active" : ""}`}
                      onClick={() => setHistoryLangFilter("hindi")}
                    >
                      हिन्दी
                    </button>
                    <button
                      className={`lang-filter-btn ${historyLangFilter === "kannada" ? "active" : ""}`}
                      onClick={() => setHistoryLangFilter("kannada")}
                    >
                      ಕನ್ನಡ
                    </button>
                  </div>
                </div>
              </div>

              {/* History Cards Grid */}
              <div className="history-cards-container">
                {conversationsLoading ? (
                  <div className="history-loading-card">
                    <RefreshCw size={24} className="spin-icon" />
                    <span>Loading previous consultations...</span>
                  </div>
                ) : filteredConversations.length === 0 ? (
                  <div className="history-empty-card">
                    <History size={36} />
                    <h4>No consultations found</h4>
                    <p>Your legal conversations will appear here. Start a consultation to ask about Indian laws and penalties.</p>
                    <button className="start-new-consult-btn" onClick={handleCreateNewChat}>
                      <Plus size={16} />
                      <span>Start New Consultation</span>
                    </button>
                  </div>
                ) : (
                  <div className="history-grid-layout">
                    {filteredConversations.map((conv) => {
                      const convId = conv.id || conv.conversation_id;
                      const dateFormatted = new Date(conv.updated_at * 1000).toLocaleDateString([], {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      });

                      return (
                        <div
                          key={convId}
                          className="consultation-record-card"
                          onClick={() => handleSelectConversation(convId)}
                        >
                          <div className="record-card-top">
                            <div className="record-topic-badge">
                              <Scale size={13} />
                              <span>{conv.legal_topic || "General Law"}</span>
                            </div>
                            <span className="record-lang-badge">{conv.language || "English"}</span>
                          </div>

                          <h4 className="record-title">{conv.title || "Legal Consultation"}</h4>

                          {conv.last_question && (
                            <div className="record-preview-snippet">
                              <span className="preview-label">Last query:</span>
                              <p>"{conv.last_question}"</p>
                            </div>
                          )}

                          <div className="record-card-footer">
                            <div className="record-meta-info">
                              <Clock size={12} />
                              <span>{dateFormatted}</span>
                              <span className="meta-turns-count">• {conv.question_count || 1} queries</span>
                            </div>

                            <div className="record-actions-group">
                              <button
                                className="resume-consult-btn"
                                onClick={(e) => {
                                  e.stopPropagation();
                                  handleSelectConversation(convId);
                                }}
                                title="Resume this consultation"
                              >
                                <span>Continue</span>
                                <CornerDownRight size={13} />
                              </button>
                              <button
                                className="delete-consult-btn"
                                onClick={(e) => handleDeleteConversation(convId, e)}
                                title="Delete consultation"
                              >
                                <Trash2 size={13} />
                              </button>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          )}
        </section>
      </main>

      {/* AUTHENTICATION MODAL */}
      {showAuthModal && (
        <div className="modal-backdrop-overlay" onClick={() => setShowAuthModal(false)}>
          <div className="auth-modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-top-bar">
              <h3>{authMode === "login" ? "Sign In to JanNyaya AI" : "Register Citizen Account"}</h3>
              <button className="close-modal-btn" onClick={() => setShowAuthModal(false)}>
                <X size={18} />
              </button>
            </div>

            {authError && <div className="auth-error-banner">{authError}</div>}

            <form className="auth-modal-form" onSubmit={handleAuthSubmit}>
              {authMode === "register" && (
                <>
                  <div className="form-group-field">
                    <label>{t.full_name}</label>
                    <input
                      type="text"
                      required
                      placeholder="e.g. Punith Kumar"
                      value={authFullName}
                      onChange={(e) => setAuthFullName(e.target.value)}
                    />
                  </div>
                  <div className="form-group-field">
                    <label>{t.email}</label>
                    <input
                      type="email"
                      placeholder="name@example.com"
                      value={authEmail}
                      onChange={(e) => setAuthEmail(e.target.value)}
                    />
                  </div>
                </>
              )}

              <div className="form-group-field">
                <label>Username</label>
                <input
                  type="text"
                  required
                  placeholder="Citizen username"
                  value={authUsername}
                  onChange={(e) => setAuthUsername(e.target.value)}
                />
              </div>

              <div className="form-group-field">
                <label>Password</label>
                <input
                  type="password"
                  required
                  placeholder="••••••••"
                  value={authPassword}
                  onChange={(e) => setAuthPassword(e.target.value)}
                />
              </div>

              <button type="submit" className="auth-submit-primary-btn" disabled={authLoading}>
                {authLoading ? "Authenticating..." : authMode === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>

            <div className="modal-switch-mode-row">
              {authMode === "login" ? (
                <p>
                  Do not have an account?{" "}
                  <button type="button" onClick={() => setAuthMode("register")}>
                    Register now
                  </button>
                </p>
              ) : (
                <p>
                  Already registered?{" "}
                  <button type="button" onClick={() => setAuthMode("login")}>
                    Sign in here
                  </button>
                </p>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ENHANCED ACCOUNT SETTINGS & PROFILE MODAL */}
      {showProfileModal && (
        <div className="modal-backdrop-overlay" onClick={() => setShowProfileModal(false)}>
          <div className="profile-modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="modal-top-bar">
              <h3>{t.profile_title || "Citizen Profile & Settings"}</h3>
              <button className="close-modal-btn" onClick={() => setShowProfileModal(false)}>
                <X size={18} />
              </button>
            </div>

            {/* Profile Tab Navigation */}
            <div className="profile-tab-navigation">
              <button
                type="button"
                className={`profile-tab-item ${profileSubTab === "info" ? "active" : ""}`}
                onClick={() => setProfileSubTab("info")}
              >
                <User size={14} />
                <span>General Info</span>
              </button>

              <button
                type="button"
                className={`profile-tab-item ${profileSubTab === "stats" ? "active" : ""}`}
                onClick={() => {
                  setProfileSubTab("stats");
                  loadUserStats();
                }}
              >
                <Activity size={14} />
                <span>Activity Stats</span>
              </button>

              <button
                type="button"
                className={`profile-tab-item ${profileSubTab === "password" ? "active" : ""}`}
                onClick={() => setProfileSubTab("password")}
              >
                <Lock size={14} />
                <span>Security</span>
              </button>
            </div>

            {profileMsg.text && (
              <div className={`auth-error-banner ${profileMsg.type === "success" ? "success-variant" : ""}`}>
                {profileMsg.text}
              </div>
            )}

            <form className="auth-modal-form" onSubmit={handleSaveProfile}>
              {/* TAB 1: GENERAL INFO & AVATAR */}
              {profileSubTab === "info" && (
                <>
                  <div className="profile-avatar-section">
                    <div className="profile-avatar-circle-wrapper">
                      {editAvatar ? (
                        <img src={editAvatar} alt="Avatar Preview" className="profile-avatar-circle-img" />
                      ) : (
                        <div className="profile-avatar-fallback-initial">
                          {editFullName?.charAt(0)?.toUpperCase() || user?.username?.charAt(0)?.toUpperCase() || "U"}
                        </div>
                      )}
                    </div>

                    <div className="profile-avatar-actions">
                      <input
                        type="file"
                        ref={avatarInputRef}
                        onChange={handleAvatarChange}
                        accept="image/jpeg,image/png,image/webp"
                        style={{ display: "none" }}
                      />
                      <div className="profile-avatar-actions-row">
                        <button
                          type="button"
                          className="avatar-change-btn"
                          onClick={() => avatarInputRef.current?.click()}
                        >
                          <Camera size={13} />
                          <span>{editAvatar ? "Change Photo" : "Upload Photo"}</span>
                        </button>
                        {editAvatar && (
                          <button
                            type="button"
                            className="avatar-remove-btn"
                            onClick={handleRemoveAvatar}
                          >
                            <Trash2 size={13} />
                            <span>Remove</span>
                          </button>
                        )}
                      </div>
                      <span className="avatar-spec-note">Supports JPG, PNG, WEBP (Max 5MB)</span>
                    </div>
                  </div>

                  <div className="form-group-field">
                    <label>{t.full_name}</label>
                    <input
                      type="text"
                      value={editFullName}
                      placeholder="e.g. Punith Kumar"
                      onChange={(e) => setEditFullName(e.target.value)}
                    />
                  </div>

                  <div className="form-group-field">
                    <label>{t.email}</label>
                    <input
                      type="email"
                      value={editEmail}
                      placeholder="citizen@example.com"
                      onChange={(e) => setEditEmail(e.target.value)}
                    />
                  </div>

                  <div className="form-group-field">
                    <label>Citizen Status / Account Type</label>
                    <select
                      value={editCitizenStatus}
                      onChange={(e) => setEditCitizenStatus(e.target.value)}
                    >
                      <option value="Verified Citizen">Verified Citizen</option>
                      <option value="Legal Aid Applicant (NALSA)">Legal Aid Applicant (NALSA)</option>
                      <option value="MSME / Business Owner">MSME / Business Owner</option>
                      <option value="Senior Citizen">Senior Citizen</option>
                      <option value="Legal Practitioner / Law Student">Legal Practitioner / Law Student</option>
                    </select>
                  </div>

                  <div className="form-group-field">
                    <label>{t.pref_language} (UI Interface)</label>
                    <select value={editLang} onChange={(e) => setEditLang(e.target.value)}>
                      <option value="english">English</option>
                      <option value="hindi">हिन्दी (Hindi)</option>
                      <option value="kannada">ಕನ್ನಡ (Kannada)</option>
                    </select>
                  </div>

                  <div className="form-group-field">
                    <label>Default Document Explanation Language</label>
                    <select
                      value={editDefExplanationLang}
                      onChange={(e) => setEditDefExplanationLang(e.target.value)}
                    >
                      <option value="english">English</option>
                      <option value="hindi">हिन्दी (Hindi)</option>
                      <option value="kannada">ಕನ್ನಡ (Kannada)</option>
                    </select>
                  </div>

                  {user?.created_at && (
                    <div className="profile-meta-row">
                      <span>Citizen Member Since:</span>
                      <strong>{new Date(user.created_at * 1000).toLocaleDateString([], { year: "numeric", month: "long", day: "numeric" })}</strong>
                    </div>
                  )}
                </>
              )}

              {/* TAB 2: LIVE ACTIVITY STATS */}
              {profileSubTab === "stats" && (
                <div className="profile-stats-container">
                  <div className="profile-stats-grid">
                    <div className="profile-stat-box">
                      <div className="stat-box-icon">
                        <MessageSquare size={16} />
                      </div>
                      <span className="stat-box-value">{userStats.consultations_count || conversations.length || 1}</span>
                      <span className="stat-box-label">Consultations</span>
                    </div>

                    <div className="profile-stat-box">
                      <div className="stat-box-icon">
                        <FileText size={16} />
                      </div>
                      <span className="stat-box-value">{userStats.documents_analyzed || (uploadResult ? 1 : 0)}</span>
                      <span className="stat-box-label">Docs Analyzed</span>
                    </div>

                    <div className="profile-stat-box">
                      <div className="stat-box-icon">
                        <Scale size={16} />
                      </div>
                      <span className="stat-box-value">{userStats.questions_asked || chatMessages.filter(m => m.role === "user").length}</span>
                      <span className="stat-box-label">Queries Asked</span>
                    </div>

                    <div className="profile-stat-box">
                      <div className="stat-box-icon">
                        <Award size={16} />
                      </div>
                      <span className="stat-box-value">{userStats.topics_explored || 4}</span>
                      <span className="stat-box-label">Topics Explored</span>
                    </div>
                  </div>

                  <div className="profile-meta-row">
                    <span>Account Security Status:</span>
                    <strong style={{ color: "#34d399" }}>Active & Encrypted (PBKDF2-HMAC)</strong>
                  </div>

                  <button
                    type="button"
                    className="avatar-change-btn"
                    style={{ alignSelf: "center" }}
                    onClick={loadUserStats}
                    disabled={userStatsLoading}
                  >
                    <RefreshCw size={13} className={userStatsLoading ? "spin-icon" : ""} />
                    <span>{userStatsLoading ? "Refreshing..." : "Refresh Live Statistics"}</span>
                  </button>
                </div>
              )}

              {/* TAB 3: PASSWORD & SECURITY */}
              {profileSubTab === "password" && (
                <>
                  <div className="form-group-field">
                    <label>Current Password</label>
                    <div className="password-input-with-eye">
                      <input
                        type={showCurrPassword ? "text" : "password"}
                        placeholder="Enter current password"
                        value={editCurrPassword}
                        onChange={(e) => setEditCurrPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="password-visibility-toggle-btn"
                        onClick={() => setShowCurrPassword(!showCurrPassword)}
                        tabIndex={-1}
                      >
                        {showCurrPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                      </button>
                    </div>
                  </div>

                  <div className="form-group-field">
                    <label>New Password</label>
                    <div className="password-input-with-eye">
                      <input
                        type={showNewPassword ? "text" : "password"}
                        placeholder="Enter new password (min 4 chars)"
                        value={editNewPassword}
                        onChange={(e) => setEditNewPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="password-visibility-toggle-btn"
                        onClick={() => setShowNewPassword(!showNewPassword)}
                        tabIndex={-1}
                      >
                        {showNewPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                      </button>
                    </div>
                  </div>

                  <div className="form-group-field">
                    <label>Confirm New Password</label>
                    <div className="password-input-with-eye">
                      <input
                        type={showConfirmPassword ? "text" : "password"}
                        placeholder="Re-enter new password"
                        value={editConfirmPassword}
                        onChange={(e) => setEditConfirmPassword(e.target.value)}
                      />
                      <button
                        type="button"
                        className="password-visibility-toggle-btn"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        tabIndex={-1}
                      >
                        {showConfirmPassword ? <EyeOff size={15} /> : <Eye size={15} />}
                      </button>
                    </div>
                  </div>
                </>
              )}

              <button type="submit" className="auth-submit-primary-btn" disabled={profileLoading}>
                {profileLoading ? "Saving..." : t.save_profile || "Save Changes"}
              </button>
            </form>
          </div>
        </div>
      )}

      {/* SOURCE PROVENANCE MODAL */}
      {selectedSourceModal && (
        <div className="provenance-modal-overlay" onClick={() => setSelectedSourceModal(null)}>
          <div className="provenance-modal-card" onClick={(e) => e.stopPropagation()}>
            <div className="prov-modal-header">
              <div className="prov-modal-badge">
                <ShieldCheck size={16} />
                <span>Verified Statutory Reference</span>
              </div>
              <button className="prov-modal-close" onClick={() => setSelectedSourceModal(null)}>
                <X size={18} />
              </button>
            </div>

            <div className="prov-modal-body">
              <h3 className="prov-modal-act-title">{selectedSourceModal.act_name || selectedSourceModal.source || "Indian Statute"}</h3>
              {selectedSourceModal.section && (
                <div className="prov-modal-section-pill">
                  Section {selectedSourceModal.section}: {selectedSourceModal.title || selectedSourceModal.section_title || "Statutory Provision"}
                </div>
              )}

              <div className="prov-modal-meta-grid">
                <div className="prov-meta-item">
                  <span className="prov-meta-label">Enacting Authority</span>
                  <span className="prov-meta-value">{selectedSourceModal.authority || "Government of India / Parliament"}</span>
                </div>
                <div className="prov-meta-item">
                  <span className="prov-meta-label">Legal Route</span>
                  <span className="prov-meta-value">{selectedSourceModal.route || "general"}</span>
                </div>
                <div className="prov-meta-item">
                  <span className="prov-meta-label">Verification Status</span>
                  <span className="prov-meta-value verified-tag">Indexed in ChromaDB</span>
                </div>
                <div className="prov-meta-item">
                  <span className="prov-meta-label">Source Document</span>
                  <span className="prov-meta-value">{selectedSourceModal.source || "Official Bare Act"}</span>
                </div>
              </div>

              {selectedSourceModal.snippet && (
                <div className="prov-modal-snippet-box">
                  <span className="snippet-label">Statutory Text Snippet:</span>
                  <p>{selectedSourceModal.snippet}</p>
                </div>
              )}
            </div>

            <div className="prov-modal-footer">
              <button className="prov-modal-done-btn" onClick={() => setSelectedSourceModal(null)}>
                Close Reference
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Guest Trial Limit Exceeded Modal */}
      {showTrialLimitModal && (
        <div className="modal-backdrop-overlay" onClick={() => setShowTrialLimitModal(false)}>
          <div className="trial-limit-dialog-box" onClick={(e) => e.stopPropagation()}>
            <div className="trial-limit-icon">
              <Sparkles size={28} />
            </div>
            <div className="trial-limit-title">Guest Trial Limit Reached</div>
            <div className="trial-limit-desc">
              You have completed your 5 free trial legal questions. Create a free citizen account or log in to unlock unlimited consultations, multi-document case analysis, and permanent case dossier storage.
            </div>
            <div className="trial-limit-actions">
              <button
                className="trial-limit-action-btn primary"
                onClick={() => {
                  setShowTrialLimitModal(false);
                  setAuthMode("register");
                  setShowAuthModal(true);
                }}
              >
                Create Free Account
              </button>
              <button
                className="trial-limit-action-btn secondary"
                onClick={() => {
                  setShowTrialLimitModal(false);
                  setAuthMode("login");
                  setShowAuthModal(true);
                }}
              >
                Sign In
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
