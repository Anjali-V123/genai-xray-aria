import { useState, useRef, useEffect, useCallback } from "react";

const HOSPITALS = ["PES Hospital", "Apollo", "Manipal", "Fortis", "AIIMS"];
const FLASK_URL = "http://localhost:5000";

const HOSPITAL_COLORS = {
  "PES Hospital": { bg: "#1A3C6E", light: "#EBF0FA", text: "#1A3C6E", accent: "#2563EB", logo: "🏛️" },
  "Apollo":       { bg: "#005BAA", light: "#E6F2FF", text: "#005BAA", accent: "#0284C7", logo: "⚕️" },
  "Manipal":      { bg: "#C8102E", light: "#FEF0F2", text: "#C8102E", accent: "#E11D48", logo: "🏥" },
  "Fortis":       { bg: "#E85D1E", light: "#FEF3EE", text: "#C24A12", accent: "#EA580C", logo: "🔶" },
  "AIIMS":        { bg: "#1E3A5F", light: "#EEF3FA", text: "#1E3A5F", accent: "#1D4ED8", logo: "🎓" },
};

// ── Onboarding ────────────────────────────────────────────────────────────────
const STEPS = [
  { title: "Welcome to Dr. ARIA", icon: "👨‍⚕️", msg: "I'm Dr. ARIA — your AI Radiology Intelligence Assistant. I analyse chest X-rays and generate detailed medical reports for 5 hospitals. Ready to get started?" },
  { title: "Step 1 · Upload X-ray", icon: "📤", msg: "Upload a chest X-ray image (PNG or JPG). Important: only real grayscale chest X-ray images are accepted. I'll reject photos, selfies, or non-X-ray images automatically." },
  { title: "Step 2 · Patient Details", icon: "📋", msg: "Optionally fill in the patient name, age, sex, and referring doctor. These appear in the generated hospital reports just like a real radiology report." },
  { title: "Step 3 · Choose Mode", icon: "🩺", msg: "Doctor Mode generates clinical radiology reports with technical terminology. Patient Mode generates simple, easy-to-understand language for the patient." },
  { title: "Step 4 · Get Reports", icon: "📊", msg: "Click Analyse X-ray. I'll run GradCAM heatmap analysis, detect conditions, and generate properly formatted reports for PES Hospital, Apollo, Manipal, Fortis, and AIIMS — each in their own style." },
  { title: "You're all set!", icon: "✅", msg: "Switch between hospital tabs to compare reports. Download individual reports or all at once. Chat with me anytime about the findings. Let's begin!" },
];

function Onboarding({ onDone }) {
  const [step, setStep] = useState(0);
  const s = STEPS[step];
  const last = step === STEPS.length - 1;
  return (
    <div style={{ position:"fixed", inset:0, zIndex:2000, background:"#0B1426", display:"flex", alignItems:"center", justifyContent:"center" }}>
      <div style={{ position:"absolute", inset:0, backgroundImage:"radial-gradient(circle at 20% 50%, rgba(37,99,235,0.08) 0%, transparent 50%), radial-gradient(circle at 80% 20%, rgba(29,158,117,0.08) 0%, transparent 50%)" }} />
      <div style={{ maxWidth:520, width:"90%", position:"relative", animation:"fadeUp 0.4s ease" }}>
        <div style={{ display:"flex", gap:6, justifyContent:"center", marginBottom:28 }}>
          {STEPS.map((_,i) => <div key={i} style={{ height:4, width: i===step?28:8, borderRadius:4, background: i<=step?"#3B82F6":"rgba(255,255,255,0.1)", transition:"all 0.3s" }} />)}
        </div>
        <div style={{ background:"rgba(255,255,255,0.04)", border:"1px solid rgba(255,255,255,0.08)", borderRadius:20, padding:"36px 40px" }}>
          <div style={{ fontSize:48, textAlign:"center", marginBottom:16 }}>{s.icon}</div>
          <h2 style={{ color:"#fff", textAlign:"center", marginBottom:14, fontSize:20, fontWeight:700 }}>{s.title}</h2>
          <p style={{ color:"rgba(255,255,255,0.65)", textAlign:"center", lineHeight:1.75, fontSize:14, marginBottom:28 }}>{s.msg}</p>
          <button onClick={() => last ? onDone() : setStep(s=>s+1)} style={{ width:"100%", padding:"13px 0", borderRadius:12, border:"none", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", color:"#fff", fontSize:15, fontWeight:600, cursor:"pointer" }}>{last ? "Start Analysing →" : "Next →"}</button>
          <button onClick={onDone} style={{ display:"block", margin:"12px auto 0", background:"none", border:"none", color:"rgba(255,255,255,0.25)", fontSize:12, cursor:"pointer" }}>Skip</button>
        </div>
      </div>
      <style>{`@keyframes fadeUp { from{opacity:0;transform:translateY(20px)} to{opacity:1;transform:translateY(0)} }`}</style>
    </div>
  );
}

// ── Shared report body renderer ───────────────────────────────────────────────
function FullReportBody({ r, accentColor }) {
  if (!r.full_report) return null;
  return (
    <>
      {r.full_report.split('\n').map((line, i) => {
        const trimmed = line.trim();
        if (!trimmed) return <div key={i} style={{ height:7 }} />;
        const isHeading = /^[A-Z][A-Z0-9\s\/\-:]{2,48}:?\s*$/.test(trimmed);
        const isBullet  = trimmed.startsWith('-') || trimmed.startsWith('•') || /^\d+\./.test(trimmed);
        if (isHeading) return <div key={i} style={{ fontWeight:700, fontSize:12, color:accentColor, marginTop:14, marginBottom:4, textTransform:"uppercase", letterSpacing:0.7, borderBottom:`1px solid ${accentColor}22`, paddingBottom:3 }}>{trimmed}</div>;
        if (isBullet)  return <div key={i} style={{ marginBottom:5, paddingLeft:14, borderLeft:`2px solid ${accentColor}44`, paddingTop:1 }}>{trimmed}</div>;
        return <div key={i} style={{ marginBottom:4, lineHeight:1.65 }}>{trimmed}</div>;
      })}
    </>
  );
}

// ── Hospital Report Renderers ─────────────────────────────────────────────────
function PESReport({ r }) {
  const c = HOSPITAL_COLORS["PES Hospital"];
  return (
    <div style={{ fontFamily:"'Times New Roman', serif", fontSize:13, color:"#111", background:"#fff", minHeight:500 }}>
      <div style={{ borderBottom:`3px solid ${c.bg}`, padding:"16px 24px 12px" }}>
        <div style={{ display:"flex", alignItems:"center", gap:14 }}>
          <div style={{ width:54, height:54, borderRadius:"50%", background:c.bg, display:"flex", alignItems:"center", justifyContent:"center", fontSize:24, color:"#fff", flexShrink:0 }}>🏛️</div>
          <div>
            <div style={{ fontSize:16, fontWeight:700, color:c.bg }}>{r.meta.full_name}</div>
            <div style={{ fontSize:10, color:"#555", marginTop:2 }}>{r.meta.address}</div>
            <div style={{ fontSize:10, color:"#555" }}>Tel: {r.meta.phone}</div>
          </div>
        </div>
        <div style={{ marginTop:10, textAlign:"center", fontSize:13, fontWeight:700, color:c.bg, letterSpacing:1, borderTop:`1px solid ${c.bg}33`, paddingTop:8 }}>CT THORAX / CHEST X-RAY REPORT</div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:0, borderBottom:"1px solid #ddd", fontSize:12 }}>
        <div style={{ padding:"8px 24px", borderRight:"1px solid #ddd" }}>
          <b>Patient ID:</b> {`PES${Date.now().toString().slice(-6)}`}<br/>
          <b>Accession No:</b> {r.accession}<br/>
          <b>Prepared By:</b> {r.meta.radiologist}
        </div>
        <div style={{ padding:"8px 24px" }}>
          <b>Patient Name:</b> {r.patient_name}<br/>
          <b>Age/Gender:</b> {r.patient_age} / {r.patient_sex}<br/>
          <b>Date:</b> {r.date}
        </div>
      </div>
      <div style={{ padding:"16px 24px" }}>
        <FullReportBody r={r} accentColor={c.bg} />
        {!r.full_report && <>
          {r.bullet_findings.map((f,i) => <div key={i} style={{ marginBottom:6, paddingLeft:12, borderLeft:`2px solid ${c.bg}44` }}><span style={{ fontWeight:600 }}>{f.condition}</span> ({f.confidence}) — {f.description}</div>)}
          <div style={{ marginTop:14, fontWeight:700, fontSize:12, textTransform:"uppercase", color:c.bg }}>IMPRESSION</div>
          <div style={{ marginTop:6, paddingLeft:12 }}>{r.impression}</div>
          <div style={{ marginTop:14, fontWeight:700, fontSize:12, textTransform:"uppercase", color:c.bg }}>{r.mode==="Patient"?"ADVICE":"RECOMMENDATIONS"}</div>
          <div style={{ marginTop:6, paddingLeft:12, fontStyle:"italic" }}>{r.advice}</div>
        </>}
        <div style={{ marginTop:24, display:"flex", justifyContent:"space-between", borderTop:"1px solid #ddd", paddingTop:12, fontSize:11 }}>
          <div><b>{r.meta.radiologist}</b></div>
          <div style={{ textAlign:"right" }}><b>{r.meta.senior}</b></div>
        </div>
      </div>
    </div>
  );
}

function ApolloReport({ r }) {
  const c = HOSPITAL_COLORS["Apollo"];
  return (
    <div style={{ fontFamily:"Arial, sans-serif", fontSize:13, color:"#111", background:"#fff", minHeight:500 }}>
      <div style={{ background:c.bg, padding:"14px 24px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div>
          <div style={{ color:"#fff", fontWeight:700, fontSize:18, letterSpacing:1 }}>APOLLO HOSPITALS</div>
          <div style={{ color:"rgba(255,255,255,0.7)", fontSize:10 }}>X-Ray | CT-Scan | MRI | USG</div>
          <div style={{ color:"rgba(255,255,255,0.6)", fontSize:10 }}>{r.meta.address}</div>
        </div>
        <div style={{ textAlign:"right", color:"rgba(255,255,255,0.8)", fontSize:11 }}>
          <div>📞 {r.meta.phone}</div>
          <div style={{ marginTop:6, background:"rgba(255,255,255,0.15)", padding:"3px 10px", borderRadius:12, fontSize:10 }}>www.apollohospitals.com</div>
        </div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", background:"#F0F7FF", padding:"10px 24px", fontSize:12, borderBottom:"1px solid #BFD9F0" }}>
        <div><b>Patient Name:</b> {r.patient_name}<br/><b>Age:</b> {r.patient_age} &nbsp;&nbsp; <b>Sex:</b> {r.patient_sex}<br/><b>Ref. By:</b> {r.ref_by}</div>
        <div style={{ textAlign:"right" }}><b>PID:</b> {`APL${Date.now().toString().slice(-5)}`}<br/><b>Date:</b> {r.date} · {r.time}<br/><b>Report Date:</b> {r.date}</div>
      </div>
      <div style={{ textAlign:"center", padding:"14px 24px 8px", borderBottom:"2px solid #BFD9F0" }}>
        <div style={{ fontWeight:700, fontSize:15, color:c.bg }}>X-RAY CHEST</div>
        <div style={{ fontSize:11, color:"#555" }}>X-Ray Chest – PA View · {r.technique}</div>
      </div>
      <div style={{ padding:"14px 24px" }}>
        <FullReportBody r={r} accentColor={c.bg} />
        {!r.full_report && <>
          {r.bullet_findings.map((f,i) => <div key={i} style={{ display:"flex", alignItems:"flex-start", gap:8, marginBottom:7 }}><span style={{ color:c.bg, fontWeight:700, marginTop:1 }}>•</span><span><b>{f.condition}</b> (conf: {f.confidence}): {f.description}</span></div>)}
          <div style={{ marginTop:14 }}><div style={{ fontWeight:700, color:c.bg, fontSize:12, textTransform:"uppercase", marginBottom:4 }}>IMPRESSION</div><div>{r.impression}</div></div>
          <div style={{ marginTop:12 }}><div style={{ fontWeight:700, color:c.bg, fontSize:12, textTransform:"uppercase", marginBottom:4 }}>{r.mode==="Patient"?"ADVICE FOR PATIENT":"CLINICAL ADVICE"}</div><div style={{ fontStyle:"italic" }}>{r.advice}</div></div>
        </>}
        <div style={{ marginTop:20, display:"flex", justifyContent:"space-between", borderTop:"1px solid #ddd", paddingTop:12, fontSize:11 }}>
          <div style={{ textAlign:"center" }}><div style={{ borderTop:"1px solid #333", width:120, paddingTop:4 }}><b>{r.meta.radiologist}</b></div></div>
          <div style={{ textAlign:"center" }}><div style={{ borderTop:"1px solid #333", width:120, paddingTop:4 }}><b>{r.meta.senior}</b></div></div>
        </div>
      </div>
    </div>
  );
}

function ManipالReport({ r }) {
  const c = HOSPITAL_COLORS["Manipal"];
  return (
    <div style={{ fontFamily:"'Segoe UI', sans-serif", fontSize:13, color:"#111", background:"#fff", minHeight:500 }}>
      <div style={{ background:"linear-gradient(135deg,#C8102E,#E11D48)", padding:"16px 24px", display:"flex", alignItems:"center", gap:14 }}>
        <div style={{ width:50, height:50, borderRadius:8, background:"rgba(255,255,255,0.2)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:22 }}>🏥</div>
        <div>
          <div style={{ color:"#fff", fontWeight:700, fontSize:17 }}>MANIPAL HOSPITAL</div>
          <div style={{ color:"rgba(255,255,255,0.75)", fontSize:10 }}>{r.meta.address} · Tel: {r.meta.phone}</div>
        </div>
      </div>
      <table style={{ width:"100%", borderCollapse:"collapse", fontSize:12 }}>
        <tbody>
          <tr style={{ background:"#FEF0F2" }}><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc", fontWeight:600, width:"25%" }}>Patient Name</td><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc" }}>{r.patient_name}</td><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc", fontWeight:600, width:"20%" }}>Date</td><td style={{ padding:"6px 16px" }}>{r.date}</td></tr>
          <tr><td style={{ padding:"6px 16px", borderRight:"1px solid #eee", fontWeight:600 }}>Age / Sex</td><td style={{ padding:"6px 16px", borderRight:"1px solid #eee" }}>{r.patient_age} / {r.patient_sex}</td><td style={{ padding:"6px 16px", borderRight:"1px solid #eee", fontWeight:600 }}>Ref. By</td><td style={{ padding:"6px 16px" }}>{r.ref_by}</td></tr>
          <tr style={{ background:"#FEF0F2" }}><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc", fontWeight:600 }}>Accession</td><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc" }}>{r.accession}</td><td style={{ padding:"6px 16px", borderRight:"1px solid #fcc", fontWeight:600 }}>Technique</td><td style={{ padding:"6px 16px", fontSize:11 }}>{r.technique}</td></tr>
        </tbody>
      </table>
      <div style={{ padding:"14px 24px" }}>
        <div style={{ fontWeight:700, fontSize:14, color:c.bg, textAlign:"center", borderBottom:`2px solid ${c.bg}`, paddingBottom:6, marginBottom:12 }}>X-RAY CHEST REPORT</div>
        <FullReportBody r={r} accentColor={c.bg} />
        {!r.full_report && <>
          <div style={{ fontWeight:700, color:c.bg, marginBottom:8 }}>FINDINGS</div>
          {r.bullet_findings.map((f,i) => <div key={i} style={{ background: i%2===0?"#FEF9FA":"#fff", padding:"6px 10px", borderLeft:`3px solid ${c.bg}`, marginBottom:5, fontSize:12 }}><b>{f.condition}</b> — {f.description} <span style={{ color:"#888", fontSize:10 }}>[{f.confidence}]</span></div>)}
          <div style={{ marginTop:14, fontWeight:700, color:c.bg }}>IMPRESSION</div>
          <div style={{ marginTop:6, background:"#FEF0F2", padding:"8px 12px", borderRadius:6 }}>{r.impression}</div>
          <div style={{ marginTop:12, fontWeight:700, color:c.bg }}>{r.mode==="Patient"?"PATIENT ADVICE":"RECOMMENDATIONS"}</div>
          <div style={{ marginTop:6, fontStyle:"italic", color:"#444" }}>{r.advice}</div>
        </>}
        <div style={{ marginTop:20, fontSize:11, display:"flex", justifyContent:"space-between", borderTop:"1px solid #fcc", paddingTop:10 }}>
          <div><b>{r.meta.radiologist}</b></div>
          <div style={{ textAlign:"right" }}><b>{r.meta.senior}</b></div>
        </div>
      </div>
    </div>
  );
}

function FortisReport({ r }) {
  const c = HOSPITAL_COLORS["Fortis"];
  return (
    <div style={{ fontFamily:"'Calibri', sans-serif", fontSize:13, color:"#111", background:"#fff", minHeight:500 }}>
      <div style={{ display:"flex", alignItems:"stretch" }}>
        <div style={{ width:8, background:c.bg, flexShrink:0 }} />
        <div style={{ flex:1, padding:"14px 20px", borderBottom:"1px solid #E5E7EB" }}>
          <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center" }}>
            <div>
              <div style={{ fontWeight:700, fontSize:18, color:c.bg }}>FORTIS HOSPITAL</div>
              <div style={{ fontSize:10, color:"#666" }}>{r.meta.address} · {r.meta.phone}</div>
            </div>
            <div style={{ textAlign:"right", fontSize:11, color:"#666" }}>
              <div><b>MRNO:</b> FOR-{Date.now().toString().slice(-8)}</div>
              <div><b>Visit:</b> {r.date} {r.time}</div>
            </div>
          </div>
        </div>
      </div>
      <div style={{ display:"flex", background:"#FEF3EE", padding:"8px 28px", fontSize:12, gap:32, borderBottom:"1px solid #FDE0CC" }}>
        <span><b>Patient:</b> {r.patient_name}</span>
        <span><b>Age/Sex:</b> {r.patient_age}/{r.patient_sex}</span>
        <span><b>Ref By:</b> {r.ref_by}</span>
      </div>
      <div style={{ padding:"14px 28px" }}>
        <div style={{ fontWeight:700, color:c.bg, textAlign:"center", fontSize:15, marginBottom:6 }}>X-RAY CHEST — FINAL REPORT</div>
        <div style={{ fontSize:11, color:"#888", textAlign:"center", marginBottom:14 }}>TECHNIQUE: {r.technique}</div>
        <FullReportBody r={r} accentColor={c.bg} />
        {!r.full_report && <>
          <div style={{ fontWeight:700, color:c.bg, marginBottom:6 }}>REPORT:</div>
          {r.bullet_findings.map((f,i) => <div key={i} style={{ marginBottom:5 }}><span style={{ color:c.bg, fontWeight:600 }}>{f.condition} ({f.confidence}):</span> {f.description}</div>)}
          <div style={{ marginTop:12 }}><span style={{ fontWeight:700, color:c.bg }}>IMPRESSION: </span>{r.impression}</div>
          <div style={{ marginTop:8 }}><span style={{ fontWeight:700, color:c.bg }}>{r.mode==="Patient"?"ADVICE: ":"RECOMMENDATION: "}</span><span style={{ fontStyle:"italic" }}>{r.advice}</span></div>
        </>}
        <div style={{ marginTop:20, display:"flex", justifyContent:"space-between", borderTop:"1px solid #FDE0CC", paddingTop:12, fontSize:11 }}>
          <div><b>{r.meta.radiologist}</b><br/>MBBS, MD Radiology</div>
          <div style={{ textAlign:"right" }}><b>{r.meta.senior}</b><br/>Chief Radiologist</div>
        </div>
      </div>
    </div>
  );
}

function AIIMSReport({ r }) {
  const c = HOSPITAL_COLORS["AIIMS"];
  return (
    <div style={{ fontFamily:"'Times New Roman', serif", fontSize:13, color:"#111", background:"#fff", minHeight:500 }}>
      <div style={{ background:c.bg, padding:"14px 24px", display:"flex", justifyContent:"space-between", alignItems:"center" }}>
        <div>
          <div style={{ color:"#fff", fontWeight:700, fontSize:16 }}>ALL INDIA INSTITUTE OF MEDICAL SCIENCES</div>
          <div style={{ color:"rgba(255,255,255,0.7)", fontSize:10 }}>Department of Radiodiagnosis & Imaging</div>
          <div style={{ color:"rgba(255,255,255,0.6)", fontSize:10 }}>{r.meta.address}</div>
        </div>
        <div style={{ color:"rgba(255,255,255,0.8)", fontSize:11, textAlign:"right" }}>Tel: {r.meta.phone}</div>
      </div>
      <div style={{ textAlign:"center", padding:"12px 24px 8px", borderBottom:"2px solid #1E3A5F" }}>
        <div style={{ fontWeight:700, fontSize:14, textTransform:"uppercase", letterSpacing:1 }}>Radiology Report</div>
      </div>
      <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:0, fontSize:12, borderBottom:"1px solid #ddd" }}>
        <div style={{ padding:"10px 24px", borderRight:"1px solid #ddd" }}>
          <table style={{ width:"100%" }}><tbody>
            <tr><td style={{ fontWeight:600, width:"40%", paddingBottom:4 }}>Patient Name:</td><td>{r.patient_name}</td></tr>
            <tr><td style={{ fontWeight:600, paddingBottom:4 }}>Gender:</td><td>{r.patient_sex}</td></tr>
            <tr><td style={{ fontWeight:600 }}>Ref. Physician:</td><td>{r.ref_by}</td></tr>
          </tbody></table>
        </div>
        <div style={{ padding:"10px 24px" }}>
          <table style={{ width:"100%" }}><tbody>
            <tr><td style={{ fontWeight:600, width:"45%", paddingBottom:4 }}>Medical Record No:</td><td>AII-{Date.now().toString().slice(-6)}</td></tr>
            <tr><td style={{ fontWeight:600, paddingBottom:4 }}>Date of Study:</td><td>{r.date}</td></tr>
            <tr><td style={{ fontWeight:600 }}>Radiologist:</td><td>{r.meta.radiologist}</td></tr>
          </tbody></table>
        </div>
      </div>
      <div style={{ padding:"14px 24px" }}>
        <div style={{ background:"#F0F4FA", padding:"8px 12px", borderRadius:6, fontSize:12, marginBottom:12 }}>
          <b>Clinical History:</b> {r.patient_age > 0 ? `${r.patient_age}-year-old ${r.patient_sex} patient.` : "Adult patient."} Chest X-ray requested for evaluation.
        </div>
        <div style={{ fontWeight:700, marginBottom:4 }}>Technique</div>
        <div style={{ marginBottom:12, paddingLeft:12 }}>{r.technique}</div>
        <FullReportBody r={r} accentColor={c.bg} />
        {!r.full_report && <>
          <div style={{ fontWeight:700, marginBottom:6 }}>Findings</div>
          {r.bullet_findings.map((f,i) => <div key={i} style={{ marginBottom:5, paddingLeft:12 }}>— <b>{f.condition}</b>: {f.description} <span style={{ color:"#888", fontSize:11 }}>(confidence: {f.confidence})</span></div>)}
          <div style={{ marginTop:12, fontWeight:700 }}>Impressions</div>
          {r.impression.split(".").filter(Boolean).map((s,i) => <div key={i} style={{ paddingLeft:12, marginTop:4 }}>{i+1}. {s.trim()}.</div>)}
          <div style={{ marginTop:12, fontWeight:700 }}>Recommendations</div>
          <div style={{ paddingLeft:12, marginTop:4, fontStyle:"italic" }}>{r.advice}</div>
        </>}
        <div style={{ marginTop:24, display:"grid", gridTemplateColumns:"1fr 1fr", gap:24, borderTop:"1px solid #ddd", paddingTop:14, fontSize:11 }}>
          <div><div><b>Radiologist's name:</b> {r.meta.radiologist}</div></div>
          <div style={{ textAlign:"right" }}><div><b>Date:</b> {r.date}</div></div>
        </div>
      </div>
    </div>
  );
}

function HospitalReport({ hospital, report }) {
  if (!report) return null;
  if (hospital === "PES Hospital") return <PESReport r={report} />;
  if (hospital === "Apollo")       return <ApolloReport r={report} />;
  if (hospital === "Manipal")      return <ManipالReport r={report} />;
  if (hospital === "Fortis")       return <FortisReport r={report} />;
  if (hospital === "AIIMS")        return <AIIMSReport r={report} />;
  return null;
}

// ── Chat Panel ────────────────────────────────────────────────────────────────
function ChatPanel({ findings, mode, imageFeatures }) {
  const firstCondition = findings ? findings.split(',')[0]?.split('(')[0]?.trim() : null;
  const suggestedQuestions = findings ? [
    firstCondition ? `What does ${firstCondition} mean?` : null,
    "Is this serious?",
    "What follow-up tests are needed?",
    mode === "Patient" ? "Explain in simple terms" : "What are the differential diagnoses?",
  ].filter(Boolean) : [];

  const welcomeLines = findings
    ? [`Analysis complete. I detected: ${findings}.`, ``, `You can ask me about any of the findings — what they mean, how serious they are, or what next steps are recommended.`]
    : [`Hi, I'm Dr. ARIA — your AI Radiology Assistant.`, ``, `Once you've uploaded and analysed a chest X-ray, ask me anything about the findings.`];

  const [msgs, setMsgs] = useState([{ role:"ai", lines: welcomeLines }]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const ref = useRef();
  useEffect(() => { ref.current?.scrollIntoView({ behavior:"smooth" }); }, [msgs]);

  const send = async (text) => {
    const q = (text || input).trim();
    if (!q || loading) return;
    setInput(""); setLoading(true);
    setMsgs(m => [...m, { role:"user", lines:[q] }]);
    try {
      const res = await fetch(`${FLASK_URL}/chat`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({ question:q, findings:findings||"", mode, image_features: imageFeatures||{} }) });
      const d = await res.json();
      setMsgs(m => [...m, { role:"ai", lines: (d.answer || "Sorry, I couldn't process that.").split('\n') }]);
    } catch { setMsgs(m => [...m, { role:"ai", lines:["Connection error. Is Flask running?"] }]); }
    setLoading(false);
  };

  return (
    <div style={{ display:"flex", flexDirection:"column", height:"100%", minHeight:500, border:"1px solid #E5E7EB", borderRadius:16, overflow:"hidden", background:"#F9FAFB" }}>
      <div style={{ background:"linear-gradient(135deg,#0B1426,#1E3A5F)", padding:"12px 16px", display:"flex", alignItems:"center", gap:10 }}>
        <div style={{ width:36, height:36, borderRadius:"50%", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:18 }}>👨‍⚕️</div>
        <div>
          <div style={{ color:"#fff", fontWeight:600, fontSize:13 }}>Dr. ARIA</div>
          <div style={{ color:"#60A5FA", fontSize:11 }}>● AI Radiology Assistant · Chest X-ray Specialist</div>
        </div>
      </div>
      <div style={{ flex:1, overflowY:"auto", padding:"16px 14px", display:"flex", flexDirection:"column", gap:10 }}>
        {msgs.map((m,i) => (
          <div key={i} style={{ display:"flex", justifyContent: m.role==="user"?"flex-end":"flex-start", gap:8, alignItems:"flex-end" }}>
            {m.role==="ai" && <div style={{ width:28, height:28, borderRadius:"50%", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14, flexShrink:0 }}>👨‍⚕️</div>}
            <div style={{ maxWidth:"78%", padding:"9px 13px", background: m.role==="user"?"linear-gradient(135deg,#1D4ED8,#3B82F6)":"#fff", color: m.role==="user"?"#fff":"#1F2937", borderRadius: m.role==="user"?"18px 18px 4px 18px":"18px 18px 18px 4px", fontSize:13, lineHeight:1.65, boxShadow:"0 1px 3px rgba(0,0,0,0.08)" }}>
              {(m.lines||[m.text||""]).map((line, li) => line === "" ? <div key={li} style={{ height:6 }} /> : <div key={li}>{line}</div>)}
            </div>
          </div>
        ))}
        {loading && <div style={{ display:"flex", gap:8, alignItems:"flex-end" }}>
          <div style={{ width:28, height:28, borderRadius:"50%", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:14 }}>👨‍⚕️</div>
          <div style={{ padding:"9px 13px", background:"#fff", borderRadius:"18px 18px 18px 4px", boxShadow:"0 1px 3px rgba(0,0,0,0.08)" }}>
            <div style={{ display:"flex", gap:4 }}>{[0,1,2].map(i => <div key={i} style={{ width:6, height:6, borderRadius:"50%", background:"#3B82F6", animation:`bounce 1.2s ease infinite`, animationDelay:`${i*0.2}s` }} />)}</div>
          </div>
        </div>}
        <div ref={ref} />
      </div>
      {suggestedQuestions.length > 0 && msgs.length <= 1 && (
        <div style={{ padding:"0 12px 8px", display:"flex", flexWrap:"wrap", gap:6 }}>
          {suggestedQuestions.map((q,i) => <button key={i} onClick={()=>send(q)} style={{ padding:"5px 11px", borderRadius:16, border:"1.5px solid #BFDBFE", background:"#EFF6FF", color:"#1D4ED8", fontSize:11, fontWeight:500, cursor:"pointer", lineHeight:1.4 }}>{q}</button>)}
        </div>
      )}
      <div style={{ padding:"10px 12px", background:"#fff", borderTop:"1px solid #E5E7EB", display:"flex", gap:8 }}>
        <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>e.key==="Enter"&&send()} placeholder={findings ? "Ask about the findings…" : "Analyse an X-ray first…"} style={{ flex:1, padding:"9px 14px", border:"1.5px solid #E5E7EB", borderRadius:24, fontSize:13, outline:"none", background:"#F9FAFB", fontFamily:"system-ui" }} />
        <button onClick={()=>send()} disabled={!input.trim()||loading} style={{ width:38, height:38, borderRadius:"50%", background: input.trim()?"linear-gradient(135deg,#1D4ED8,#3B82F6)":"#E5E7EB", border:"none", cursor: input.trim()?"pointer":"not-allowed", display:"flex", alignItems:"center", justifyContent:"center", flexShrink:0 }}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
        </button>
      </div>
      <style>{`@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-5px)}}`}</style>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [onboard, setOnboard] = useState(true);
  const [mode, setMode] = useState("Doctor");
  const [imageDataUrl, setImageDataUrl] = useState(null);
  const [imageFile, setImageFile] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [reports, setReports] = useState({});
  const [findingsList, setFindingsList] = useState([]);
  const [findingsStr, setFindingsStr] = useState("");
  const [heatmapUrl, setHeatmapUrl] = useState(null);
  const [imageFeatures, setImageFeatures] = useState(null);
  const [activeTab, setActiveTab] = useState("PES Hospital");
  const [activePanel, setActivePanel] = useState("report");
  const [error, setError] = useState(null);
  const [backendStatus, setBackendStatus] = useState("checking");
  const [patientName, setPatientName] = useState("");
  const [patientAge, setPatientAge] = useState("");
  const [patientSex, setPatientSex] = useState("");
  const [refBy, setRefBy] = useState("");
  const fileRef = useRef();
  const reportRef = useRef();

  useEffect(() => {
    fetch(`${FLASK_URL}/health`).then(r=>r.json()).then(()=>setBackendStatus("online")).catch(()=>setBackendStatus("offline"));
  }, []);

  const handleFile = useCallback(file => {
    if (!file || !file.type.startsWith("image/")) return;
    setImageFile(file);
    const reader = new FileReader();
    reader.onload = e => { setImageDataUrl(e.target.result); setReports({}); setFindingsList([]); setFindingsStr(""); setHeatmapUrl(null); setError(null); setImageFeatures(null); };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(e => { e.preventDefault(); setIsDragging(false); handleFile(e.dataTransfer.files[0]); }, [handleFile]);

  const analyse = async () => {
    if (!imageFile || isGenerating) return;
    setIsGenerating(true); setError(null); setReports({}); setHeatmapUrl(null); setFindingsList([]); setFindingsStr(""); setImageFeatures(null);
    try {
      const fd = new FormData();
      fd.append("image", imageFile);
      fd.append("mode", mode);
      fd.append("patient_name", patientName || "Patient");
      fd.append("patient_age", patientAge || "--");
      fd.append("patient_sex", patientSex || "--");
      fd.append("ref_by", refBy || "Self");
      const res = await fetch(`${FLASK_URL}/analyze`, { method:"POST", body:fd });
      const data = await res.json();
      if (data.error === "not_xray") { setError("❌ " + data.message); setIsGenerating(false); return; }
      if (data.error) throw new Error(data.error);
      setFindingsList(data.findings_list || []);
      setFindingsStr(data.findings || "");
      setHeatmapUrl(`data:image/png;base64,${data.heatmap}`);
      setReports(data.reports || {});
      setImageFeatures(data.image_features || null);
    } catch(e) { setError(e.message); }
    setIsGenerating(false);
  };

  const downloadReport = (hospital) => {
    const r = reports[hospital];
    if (!r) return;
    const header = [`${"=".repeat(70)}`, `${r.meta.full_name}`, `${r.meta.address}   |   Tel: ${r.meta.phone}`, `${"=".repeat(70)}`, `CHEST X-RAY REPORT  —  AI Assisted (Dr. ARIA)`, `${"─".repeat(70)}`, `Patient : ${r.patient_name}       Age: ${r.patient_age}       Sex: ${r.patient_sex}`, `Date    : ${r.date} ${r.time}     Accession: ${r.accession}`, `Ref. By : ${r.ref_by}`, `Mode    : ${r.mode}`, `${"─".repeat(70)}`, ``];
    const body = r.full_report ? [r.full_report] : [`FINDINGS:`, ...(r.bullet_findings||[]).map(f => `  • ${f.condition} (${f.confidence}): ${f.description}`), ``, `IMPRESSION:`, `  ${r.impression}`, ``, `${r.mode==="Patient"?"ADVICE":"RECOMMENDATIONS"}:`, `  ${r.advice}`];
    const footer = [``, `${"─".repeat(70)}`, `Radiologist : ${r.meta.radiologist}`, `Senior      : ${r.meta.senior}`, `${"=".repeat(70)}`, `Generated by Dr. ARIA — AI Radiology Intelligence Assistant`, `DISCLAIMER: AI-generated report. Must be verified by a qualified radiologist before clinical use.`];
    const blob = new Blob([[...header,...body,...footer].join("\n")], { type:"text/plain" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${hospital.replace(/\s+/g,"_")}_XRay_Report_${r.date.replace(/\s/g,"_")}.txt`;
    a.click();
  };

  const downloadAll = () => HOSPITALS.forEach(h => reports[h] && downloadReport(h));
  const hasReports = Object.keys(reports).length > 0;
  const col = HOSPITAL_COLORS[activeTab];

  if (onboard) return <Onboarding onDone={() => setOnboard(false)} />;

  return (
    <div style={{ minHeight:"100vh", background:"#F1F5F9", fontFamily:"system-ui,-apple-system,sans-serif", color:"#111" }}>
      {/* Navbar */}
      <div style={{ background:"linear-gradient(135deg,#0B1426,#1E3A5F)", padding:"0 24px", height:56, display:"flex", alignItems:"center", gap:12, boxShadow:"0 2px 16px rgba(0,0,0,0.25)", position:"sticky", top:0, zIndex:100 }}>
        <div style={{ width:32, height:32, borderRadius:8, background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", display:"flex", alignItems:"center", justifyContent:"center" }}>
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5" strokeLinecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        </div>
        <div>
          <div style={{ color:"#fff", fontWeight:700, fontSize:14 }}>AI Chest X-ray Diagnosis</div>
          <div style={{ color:"rgba(255,255,255,0.35)", fontSize:10 }}>Dr. ARIA · RAG + GradCAM · Multi-Hospital</div>
        </div>
        <div style={{ marginLeft:"auto", display:"flex", gap:4, background:"rgba(255,255,255,0.07)", padding:4, borderRadius:24 }}>
          {["Doctor","Patient"].map(m => (
            <button key={m} onClick={()=>setMode(m)} style={{ padding:"4px 14px", borderRadius:20, border:"none", background: mode===m?(m==="Doctor"?"#1D4ED8":"#0F6E56"):"transparent", color: mode===m?"#fff":"rgba(255,255,255,0.4)", fontSize:12, fontWeight: mode===m?600:400, cursor:"pointer" }}>
              {m==="Doctor"?"🩺 Doctor":"👤 Patient"}
            </button>
          ))}
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:6, padding:"4px 12px", borderRadius:20, background:"rgba(255,255,255,0.07)", fontSize:11, color:"rgba(255,255,255,0.55)" }}>
          <div style={{ width:6, height:6, borderRadius:"50%", background: backendStatus==="online"?"#22C55E":backendStatus==="offline"?"#EF4444":"#F59E0B", boxShadow: backendStatus==="online"?"0 0 6px #22C55E":"none" }} />
          Flask {backendStatus}
        </div>
        <button onClick={()=>setOnboard(true)} style={{ background:"rgba(255,255,255,0.07)", border:"1px solid rgba(255,255,255,0.1)", color:"rgba(255,255,255,0.55)", padding:"4px 12px", borderRadius:20, cursor:"pointer", fontSize:11 }}>? Help</button>
      </div>

      {backendStatus==="offline" && <div style={{ background:"#FEF2F2", borderBottom:"1px solid #FECACA", padding:"8px 24px", fontSize:12, color:"#991B1B" }}>⚠️ Flask offline — run: <code style={{ background:"#fff", padding:"1px 6px", borderRadius:4 }}>cd backend && python app.py</code></div>}

      <div style={{ display:"grid", gridTemplateColumns:"280px 1fr 300px", gap:16, padding:"16px 20px", maxWidth:1440, margin:"0 auto" }}>

        {/* LEFT PANEL */}
        <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
          {/* Upload */}
          <div onClick={()=>fileRef.current?.click()} onDragOver={e=>{e.preventDefault();setIsDragging(true);}} onDragLeave={()=>setIsDragging(false)} onDrop={handleDrop}
            style={{ border: isDragging?"2px dashed #3B82F6":"2px dashed #CBD5E1", borderRadius:14, background: isDragging?"#EFF6FF":"#fff", padding: imageDataUrl?8:"24px 14px", cursor:"pointer", textAlign: imageDataUrl?"unset":"center", minHeight:210, display:"flex", flexDirection:"column", alignItems: imageDataUrl?"stretch":"center", justifyContent: imageDataUrl?"stretch":"center", position:"relative", overflow:"hidden", boxShadow:"0 1px 6px rgba(0,0,0,0.06)" }}>
            {imageDataUrl ? (
              <><img src={imageDataUrl} alt="X-ray" style={{ width:"100%", borderRadius:8, objectFit:"contain", maxHeight:240, display:"block" }} /><div style={{ position:"absolute", bottom:8, right:8, background:"rgba(0,0,0,0.6)", color:"#fff", fontSize:10, padding:"2px 8px", borderRadius:16 }}>Click to change</div></>
            ) : (
              <><div style={{ width:48, height:48, borderRadius:12, background:"#EFF6FF", border:"1px solid #BFDBFE", display:"flex", alignItems:"center", justifyContent:"center", margin:"0 auto 12px" }}>
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#3B82F6" strokeWidth="1.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/></svg>
              </div>
              <p style={{ margin:"0 0 4px", fontSize:13, fontWeight:600, color:"#374151" }}>Upload Chest X-ray</p>
              <p style={{ margin:0, fontSize:11, color:"#9CA3AF" }}>PNG or JPG · Grayscale X-ray only</p></>
            )}
          </div>
          <input ref={fileRef} type="file" accept="image/*" style={{ display:"none" }} onChange={e=>handleFile(e.target.files[0])} />

          {/* Patient details */}
          <div style={{ background:"#fff", borderRadius:14, padding:"12px 14px", boxShadow:"0 1px 6px rgba(0,0,0,0.06)", border:"1px solid #E5E7EB" }}>
            <div style={{ fontSize:10, fontWeight:700, color:"#6B7280", textTransform:"uppercase", letterSpacing:1, marginBottom:10 }}>Patient Details (Optional)</div>
            {[["Patient Name", patientName, setPatientName, "e.g. Yashvi Patel"],["Age", patientAge, setPatientAge, "e.g. 21"],["Sex", null, setPatientSex, null],["Referred By", refBy, setRefBy, "e.g. Dr. Sharma"]].map(([label, val, setter, ph], i) => (
              <div key={i} style={{ marginBottom:8 }}>
                <div style={{ fontSize:10, color:"#6B7280", marginBottom:3 }}>{label}</div>
                {label === "Sex" ? (
                  <select value={patientSex} onChange={e=>setter(e.target.value)} style={{ width:"100%", padding:"6px 10px", border:"1px solid #E5E7EB", borderRadius:8, fontSize:12, background:"#F9FAFB", color:"#374151" }}>
                    <option value="">Select</option><option value="Male">Male</option><option value="Female">Female</option><option value="Other">Other</option>
                  </select>
                ) : (
                  <input value={val} onChange={e=>setter(e.target.value)} placeholder={ph} style={{ width:"100%", padding:"6px 10px", border:"1px solid #E5E7EB", borderRadius:8, fontSize:12, background:"#F9FAFB", outline:"none", boxSizing:"border-box" }} />
                )}
              </div>
            ))}
          </div>

          {/* Heatmap */}
          <div style={{ background:"#111", borderRadius:14, overflow:"hidden", boxShadow:"0 1px 6px rgba(0,0,0,0.12)" }}>
            <div style={{ padding:"8px 12px", display:"flex", alignItems:"center", gap:7, background:"#1a1a2e" }}>
              <div style={{ width:7, height:7, borderRadius:"50%", background: heatmapUrl?"#EF4444":"#4B5563", boxShadow: heatmapUrl?"0 0 6px #EF4444":"none" }} />
              <span style={{ fontSize:11, fontWeight:600, color:"#9CA3AF" }}>GradCAM Heatmap</span>
              {heatmapUrl && <span style={{ marginLeft:"auto", fontSize:9, color:"#6B7280" }}>Affected regions</span>}
            </div>
            <div style={{ minHeight:170, display:"flex", alignItems:"center", justifyContent:"center", padding: heatmapUrl?6:16 }}>
              {heatmapUrl ? <img src={heatmapUrl} alt="Heatmap" style={{ width:"100%", borderRadius:6, objectFit:"contain", maxHeight:200 }} /> : <p style={{ fontSize:11, color:"#4B5563", margin:0, textAlign:"center" }}>{isGenerating?"Generating…":"Heatmap appears after analysis"}</p>}
            </div>
            {heatmapUrl && <div style={{ display:"flex", gap:10, justifyContent:"center", padding:"6px 0 8px", background:"#111" }}>
              {[["#0030AA","Low"],["#EF9F27","Med"],["#E24B4A","High"]].map(([c,l])=>(
                <div key={l} style={{ display:"flex", alignItems:"center", gap:4 }}><div style={{ width:9, height:9, borderRadius:2, background:c }} /><span style={{ fontSize:10, color:"#6B7280" }}>{l}</span></div>
              ))}
            </div>}
          </div>

          {/* Findings chips */}
          {findingsList.length > 0 && (
            <div style={{ background:"#fff", borderRadius:14, padding:"10px 12px", border:"1px solid #E5E7EB", boxShadow:"0 1px 6px rgba(0,0,0,0.06)" }}>
              <div style={{ fontSize:10, fontWeight:700, color:"#6B7280", textTransform:"uppercase", letterSpacing:1, marginBottom:8 }}>🔬 Detected Conditions</div>
              <div style={{ display:"flex", flexWrap:"wrap", gap:5 }}>
                {findingsList.map((f,i) => <span key={i} style={{ background:"#EFF6FF", border:"1px solid #BFDBFE", color:"#1D4ED8", fontSize:10, padding:"3px 9px", borderRadius:20, fontWeight:500 }}>{f.name} <span style={{ color:"#60A5FA" }}>{f.confidence}</span></span>)}
              </div>
            </div>
          )}

          {error && (
            <div style={{ background:"#FEF2F2", border:"1px solid #FECACA", borderRadius:12, padding:"12px 14px", lineHeight:1.6 }}>
              <div style={{ fontSize:13, fontWeight:700, color:"#991B1B", marginBottom:4 }}>⛔ Image Rejected</div>
              <div style={{ fontSize:11, color:"#B91C1C" }}>{error.replace("❌ ","")}</div>
              <div style={{ fontSize:10, color:"#EF4444", marginTop:6, fontStyle:"italic" }}>Please upload a real PA chest X-ray radiograph.</div>
            </div>
          )}

          <button onClick={analyse} disabled={!imageFile||isGenerating||backendStatus==="offline"} style={{ padding:"12px 0", borderRadius:12, border:"none", background: (!imageFile||isGenerating||backendStatus==="offline")?"#E5E7EB":"linear-gradient(135deg,#1D4ED8,#3B82F6)", color: (!imageFile||isGenerating)?"#9CA3AF":"#fff", fontSize:14, fontWeight:700, cursor: (!imageFile||isGenerating)?"not-allowed":"pointer", boxShadow: (!imageFile||isGenerating)?"none":"0 4px 16px rgba(29,78,216,0.35)" }}>
            {isGenerating ? <span style={{ display:"flex", alignItems:"center", justifyContent:"center", gap:8 }}><span style={{ width:14, height:14, border:"2px solid rgba(255,255,255,0.3)", borderTopColor:"#fff", borderRadius:"50%", animation:"spin 0.8s linear infinite", display:"inline-block" }} />Analysing…</span> : "🔍 Analyse X-ray"}
          </button>
        </div>

        {/* MIDDLE PANEL */}
        <div style={{ display:"flex", flexDirection:"column", gap:12, minWidth:0 }}>
          {/* Panel tabs — only 2 tabs now (Hospital Reports + Chat) */}
          <div style={{ display:"flex", gap:4, background:"#fff", padding:4, borderRadius:14, boxShadow:"0 1px 6px rgba(0,0,0,0.06)" }}>
            {[["report","📋 Hospital Reports"],["chat","💬 Chat with Dr. ARIA"]].map(([id,label])=>(
              <button key={id} onClick={()=>setActivePanel(id)} style={{ flex:1, padding:"8px 0", borderRadius:10, border:"none", background: activePanel===id?"linear-gradient(135deg,#0B1426,#1E3A5F)":"transparent", color: activePanel===id?"#fff":"#6B7280", fontSize:12, fontWeight: activePanel===id?600:400, cursor:"pointer" }}>{label}</button>
            ))}
          </div>

          {activePanel === "report" ? (<>
            {/* Hospital tabs */}
            <div style={{ display:"flex", gap:6, flexWrap:"wrap" }}>
              {HOSPITALS.map(h => {
                const c = HOSPITAL_COLORS[h]; const isActive = activeTab===h;
                return <button key={h} onClick={()=>setActiveTab(h)} style={{ padding:"6px 16px", borderRadius:22, border: isActive?"none":"1.5px solid #E2E8F0", background: isActive?c.bg:"#fff", color: isActive?"#fff":"#374151", fontSize:12, fontWeight: isActive?600:400, cursor:"pointer", boxShadow: isActive?`0 3px 10px ${c.bg}44`:"none", display:"flex", alignItems:"center", gap:6 }}>
                  {reports[h] && <span style={{ width:6, height:6, borderRadius:"50%", background: isActive?"rgba(255,255,255,0.7)":c.bg }} />}
                  {c.logo} {h}
                </button>;
              })}
            </div>

            {/* Report display */}
            <div ref={reportRef} style={{ background:"#fff", borderRadius:16, boxShadow:"0 2px 16px rgba(0,0,0,0.08)", border:`1px solid ${col.bg}33`, overflow:"hidden", minHeight:460 }}>
              {reports[activeTab] ? (
                <div style={{ display:"flex", flexDirection:"column" }}>
                  <div style={{ padding:"8px 16px", background:"#F8FAFC", borderBottom:"1px solid #E2E8F0", display:"flex", justifyContent:"flex-end", gap:8 }}>
                    <button onClick={()=>downloadReport(activeTab)} style={{ display:"flex", alignItems:"center", gap:6, padding:"5px 14px", background:col.bg, color:"#fff", border:"none", borderRadius:20, fontSize:11, fontWeight:600, cursor:"pointer" }}>
                      <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
                      Download This Report
                    </button>
                  </div>
                  <HospitalReport hospital={activeTab} report={reports[activeTab]} />
                </div>
              ) : (
                <div style={{ flex:1, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", padding:48, textAlign:"center", minHeight:400 }}>
                  {isGenerating ? (
                    <><div style={{ width:44, height:44, borderRadius:"50%", border:`3px solid ${col.bg}33`, borderTopColor:col.bg, animation:"spin 0.8s linear infinite", marginBottom:16 }} /><p style={{ fontSize:14, color:"#6B7280" }}>Generating {activeTab} report…</p></>
                  ) : (
                    <><div style={{ fontSize:40, marginBottom:14 }}>{col.logo}</div><p style={{ fontWeight:600, fontSize:15, margin:"0 0 6px", color:"#374151" }}>{activeTab} Report</p><p style={{ fontSize:12, color:"#9CA3AF", margin:0 }}>Upload a chest X-ray and click Analyse</p></>
                  )}
                </div>
              )}
            </div>

            {/* Mini hospital grid */}
            {hasReports && <div style={{ display:"grid", gridTemplateColumns:"repeat(5,1fr)", gap:6 }}>
              {HOSPITALS.map(h => { const c=HOSPITAL_COLORS[h]; const r=reports[h];
                return <button key={h} onClick={()=>setActiveTab(h)} style={{ padding:"8px 8px", borderRadius:10, border: activeTab===h?`2px solid ${c.bg}`:"1px solid #E5E7EB", background: activeTab===h?c.light:"#fff", textAlign:"left", cursor:"pointer" }}>
                  <div style={{ fontSize:14, marginBottom:3 }}>{c.logo}</div>
                  <div style={{ fontSize:10, fontWeight:700, color:c.text }}>{h}</div>
                  <div style={{ fontSize:9, color:"#9CA3AF", marginTop:2 }}>{r ? `${r.bullet_findings?.length||0} findings` : "Pending"}</div>
                </button>;
              })}
            </div>}
          </>) : (
            <div style={{ flex:1, minHeight:540 }}><ChatPanel findings={findingsStr} mode={mode} imageFeatures={imageFeatures} /></div>
          )}
        </div>

        {/* RIGHT PANEL */}
        <div style={{ display:"flex", flexDirection:"column", gap:12 }}>
          <div style={{ background:"linear-gradient(135deg,#0B1426,#1E3A5F)", borderRadius:16, padding:"18px 16px", boxShadow:"0 4px 16px rgba(0,0,0,0.25)" }}>
            <div style={{ display:"flex", alignItems:"center", gap:10, marginBottom:12 }}>
              <div style={{ width:44, height:44, borderRadius:"50%", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", display:"flex", alignItems:"center", justifyContent:"center", fontSize:22, border:"2px solid rgba(59,130,246,0.4)" }}>👨‍⚕️</div>
              <div><div style={{ color:"#fff", fontWeight:700, fontSize:14 }}>Dr. ARIA</div><div style={{ color:"#60A5FA", fontSize:11 }}>AI Radiology Assistant</div></div>
            </div>
            <p style={{ color:"rgba(255,255,255,0.6)", fontSize:12, lineHeight:1.7, margin:"0 0 12px" }}>
              {hasReports ? `Analysis complete! ${findingsList.length} conditions detected. ${Object.keys(reports).length} hospital reports generated in their respective formats.` : isGenerating ? "Running ResNet50 analysis and GradCAM heatmap…" : "Upload a chest X-ray to begin. I'll detect conditions and generate reports in each hospital's official format."}
            </p>
            <button onClick={()=>setActivePanel("chat")} style={{ width:"100%", padding:"8px 0", background:"rgba(59,130,246,0.15)", border:"1px solid rgba(59,130,246,0.3)", color:"#60A5FA", borderRadius:10, fontSize:12, fontWeight:600, cursor:"pointer" }}>💬 Chat with Dr. ARIA</button>
          </div>

          {hasReports && <>
            <div style={{ background:"#fff", borderRadius:14, padding:"14px", boxShadow:"0 1px 6px rgba(0,0,0,0.06)", border:"1px solid #E5E7EB" }}>
              <div style={{ fontSize:10, fontWeight:700, color:"#6B7280", textTransform:"uppercase", letterSpacing:1, marginBottom:10 }}>Analysis Summary</div>
              {[
                ["Conditions Detected", findingsList.length > 0 ? `${findingsList.length} finding${findingsList.length > 1 ? 's' : ''}` : "None"],
                ["Reports Generated", Object.keys(reports).length],
                ["Analysis Mode", mode],
                ["Technique", "PA View · ResNet50"],
              ].map(([l,v])=>(
                <div key={l} style={{ display:"flex", justifyContent:"space-between", padding:"7px 0", borderBottom:"1px solid #F3F4F6", fontSize:12 }}>
                  <span style={{ color:"#6B7280" }}>{l}</span>
                  <span style={{ fontWeight:600, color:"#111" }}>{v}</span>
                </div>
              ))}
            </div>
            <button onClick={downloadAll} style={{ padding:"11px 0", borderRadius:12, border:"none", background:"linear-gradient(135deg,#1D4ED8,#3B82F6)", color:"#fff", fontSize:13, fontWeight:600, cursor:"pointer", boxShadow:"0 4px 14px rgba(29,78,216,0.35)", display:"flex", alignItems:"center", justifyContent:"center", gap:8, width:"100%" }}>
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5"><path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/></svg>
              Download All 5 Reports
            </button>
          </>}

          <div style={{ background:"#FFFBEB", borderRadius:14, padding:"12px 14px", border:"1px solid #FDE68A" }}>
            <div style={{ fontSize:10, fontWeight:700, color:"#92400E", marginBottom:8, textTransform:"uppercase", letterSpacing:0.8 }}>💡 Tips</div>
            {["Only upload real grayscale chest X-ray images — face photos will be rejected","Fill patient details so they appear in the formatted report","Each hospital report has its own unique format and style","Heatmap highlights regions of concern within the lung field","Reports show up to 4 clinically significant findings above 65% confidence","Ask Dr. ARIA to explain any finding in plain language"].map((t,i,arr)=>(
              <div key={i} style={{ fontSize:11, color:"#78350F", lineHeight:1.5, padding:"4px 0", borderBottom: i<arr.length-1?"1px solid #FDE68A":"none" }}>• {t}</div>
            ))}
          </div>
        </div>
      </div>
      <style>{`@keyframes spin{to{transform:rotate(360deg)}} *{box-sizing:border-box}`}</style>
    </div>
  );
}
