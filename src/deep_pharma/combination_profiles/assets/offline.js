"use strict";
// Snapshot adapter only: no individual profiles, inferred probabilities or network access.
function createOfflineStore(data) {
  const key=(a,b)=>[a,b].sort().join("\x1f");
  const empty=()=>({value:null,kind:"missing",source:""});
  const finite=x=>typeof x==="number" && Number.isFinite(x) && x>=0;
  const detected=(c,t,n)=>c.value==null || (n && (c.n==null || c.n<n)) ? null : c.kind==="prr" ? c.value>t : c.value>=t;
  const pharma=(mid,i)=>{
    const m=data.molecules[mid],row=data.rows[i];
    const side=row.ids[0]===mid?0:1;
    const fields=Object.fromEntries(data.field_names.map((f,j)=>[f,data.text_pool[data.texts[i][side][j]]]));
    return {...m,fields,text_source:row.pair_id,name:null,pharmacological_class:null};
  };
  const compare=request=>{
    const a=request.a,b=request.b,source=request.source || "binary";
    if(!data.catalog.partners[a]?.includes(b))throw new Error("B doit être un partenaire de A présent dans le train.");
    if(!["binary","prr","oof"].includes(source))throw new Error("Source non incluse dans ce HTML autonome.");
    const pair=data.pairs[key(a,b)],matches=pair.rows;
    const i=request.pair_id ? matches.find(j=>data.rows[j].pair_id===request.pair_id) : matches[0];
    if(i==null)throw new Error("Pair_ID incompatible avec le couple.");
    const row=data.rows[i],kind=source==="binary"?"annotation":source==="prr"?"prr":"probability";
    const t=request.threshold===undefined?(kind==="prr"?0:.5):request.threshold;
    const margin=request.margin===undefined?.05:request.margin;
    const minimumN=request.min_support===undefined?0:request.min_support;
    if(!finite(t)||!finite(margin)||!finite(minimumN)||(kind!=="prr" && t>1))throw new Error("Seuil/marge/support invalides.");
    const support=Math.trunc(minimumN),context=!!request.context && source==="binary";
    const contextual=(mid,e)=>{
      const vote=pair.vote[e],n=data.totals[mid].n[e]-(vote==null?0:1),k=data.totals[mid].k[e]-(vote??0);
      return {value:n?k/n:null,kind:"context_frequency",n,positives:k,
        source:"Autres couples canoniques du train ; couple étudié exclu",comparability_id:"train_context",
        meaning:"pair annotation frequency",population:"train pairs",period:"dataset unspecified"};
    };
    let rows=data.effects.map(({id:effect,label},e)=>{
      const ca=context?contextual(a,e):empty(),cb=context?contextual(b,e):empty();
      const value=source==="binary"?row.binary[e]:source==="prr"?row.prr[e]:row.oof?.[e]??null;
      const ab={value,kind,source:`${source} : ${row.pair_id}`,prediction:source==="oof",row_index:i,n:null,ci:null};
      const detection=[ca,cb,ab].map(c=>detected(c,t,support));const categories=[];
      if(context && detection[0]!=null && detection[1]!=null){
        if(detection[0] && detection[1])categories.push("context_shared");
        else if(detection[0])categories.push("context_a");else if(detection[1])categories.push("context_b");
      }
      return {effect,label,a:ca,b:cb,ab,reference:null,delta:null,
        reason:context && ca.value!=null && cb.value!=null && value!=null?"Annotations/fréquences de couples : pas de probabilités individuelles.":"Profils individuels ou profil du couple manquants.",
        categories,detected:detection,prr_deltas:null,prr_log_ratios:null,soc:null,pt:null};
    });
    const f=request.filters || {};
    rows=rows.filter(r=>{
      if(!`${r.effect} ${r.label}`.toLowerCase().includes((f.search || "").toLowerCase()))return false;
      if(f.category && !r.categories.includes(f.category))return false;
      const v=r.ab.value;
      if(f.minimum!=null && (v==null || v<f.minimum))return false;
      if(f.maximum!=null && (v==null || v>f.maximum))return false;
      // No comparable monotherapy in this snapshot: no deltas or clinical metadata.
      if(f.delta_sign || f.soc || f.pt || f.minimum_n || f.max_ci_width!=null)return false;
      return true;
    });
    rows.sort(f.sort==="value"?(x,y)=>(y.ab.value??0)-(x.ab.value??0):f.sort==="delta"?()=>0:(x,y)=>x.label<y.label?-1:x.label>y.label?1:0);
    const categories={};rows.forEach(r=>r.categories.forEach(c=>categories[c]=(categories[c]||0)+1));
    const counts=[0,1,2].map(i=>rows.length && rows.every(r=>r.detected[i]!=null)?rows.reduce((s,r)=>s+Number(r.detected[i]),0):null);
    const known=context && rows.length && rows.every(r=>r.detected[0]!=null && r.detected[1]!=null);
    const union=known?rows.filter(r=>r.detected[0]||r.detected[1]).length:0;
    const shared=known?rows.filter(r=>r.detected[0]&&r.detected[1]).length:0;
    const pa=pharma(a,i),pb=pharma(b,i);
    const cyp=m=>new Set([...`${m.fields.CYP450_Enzymes} ${m.fields.Metabolism}`.toUpperCase().matchAll(/\bCYP\s*([1-4][A-Z]\d+)\b/g)].map(m=>m[1]));
    const cb=cyp(pb);
    return {rows,summary:{rows:rows.length,detected:counts,categories,jaccard:union?shared/union:null,
      jaccard_reason:"Sur effets filtrés, au seuil affiché ; contexte ≠ monothérapie.",top_positive:[],top_negative:[]},
      pharmacology:{a:pa,b:pb,shared_cyp_mentions:[...cyp(pa)].filter(c=>cb.has(c)).sort()},
      selection:{a,b,source,context,pair_id:row.pair_id,available_pairs:matches.map(j=>data.rows[j].pair_id),threshold:t,margin,min_support:minimumN},
      provenance:data.catalog.audit,limitations:data.limitations};
  };
  return {get:async(path,options={})=>{
    if(path==="/api/catalog")return data.catalog;
    if(path==="/api/compare")return compare(JSON.parse(options.body));
    throw new Error("Route indisponible hors ligne.");
  },compare};
}
async function loadOfflineSnapshot() {
  if(typeof DecompressionStream==="undefined")throw new Error("Ouvrez ce HTML avec une version récente de Chrome, Edge ou Firefox (décompression gzip requise).");
  const encoded=document.getElementById("deep-pharma-snapshot").textContent.trim();
  const bytes=Uint8Array.from(atob(encoded),c=>c.charCodeAt(0));
  const stream=new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
  return createOfflineStore(JSON.parse(await new Response(stream).text()));
}
globalThis.DeepPharmaOffline={get:async(path,options)=>{
  // Lazy initialization, shared promise: one decompression even with concurrent requests.
  globalThis.DeepPharmaOffline.ready ||= loadOfflineSnapshot();
  return (await globalThis.DeepPharmaOffline.ready).get(path,options);
}};
