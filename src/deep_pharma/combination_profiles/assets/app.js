"use strict";
// All visualizations and exports consume the exact same server-filtered rows.
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const CATEGORIES = {shared:"Partagé",a_specific:"A spécifique",b_specific:"B spécifique",strengthened:"Renforcé",attenuated:"Atténué",emerging:"Émergent",context_shared:"Contextes communs",context_a:"Contexte A seulement",context_b:"Contexte B seulement"};
const FIELDS = {Mechanism:"Mécanisme d’action",Pharmacodynamics:"Pharmacodynamie",Metabolism:"Métabolisme",Half_Life:"Demi-vie",Absorption:"Absorption",Toxicity:"Toxicité",Protein_Binding:"Liaison protéique",Elimination_Route:"Élimination",Indication:"Indications",Warning:"Avertissements",CYP450_Enzymes:"Enzymes CYP450"};
const COLORS = {a:"#176c86",b:"#a94813",ab:"#7146ba"};
let catalog, result, view = "bars", selected = null, generation = 0, debounce;
const structureURLs = new Map();
let structureBaseWidth = 640;
const num = id => $(id).value === "" ? null : Number($(id).value);
const format = (v, digits=3) => v == null ? "Non calculable" : Number(v).toLocaleString("fr-FR",{maximumFractionDigits:digits});
const formatCell = c => c.value == null ? "Non calculable" : c.kind === "context_frequency" || c.kind === "probability" ? `${format(c.value*100,1)} %` : format(c.value);

async function fetchJSON(path, options={}) {
  if (globalThis.DeepPharmaOffline) return globalThis.DeepPharmaOffline.get(path, options);
  const response = await fetch(path, options);
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || "Erreur du service local");
  return payload;
}
function partnerIds(a) { return catalog.partners?.[a] || []; }
function autocomplete(query="", side="a") {
  const q = query.toLowerCase();
  let allowed = null;
  if (side === "b") {
    try { allowed = new Set(partnerIds(resolve($("molecule-a").value))); }
    catch { allowed = new Set(); }
  }
  const matches = catalog.molecules.filter(m => (!allowed || allowed.has(m.id)) && `${m.id} ${m.smiles} ${m.search}`.toLowerCase().includes(q));
  $("molecules-"+side).innerHTML = matches.map(m => `<option value="${esc(m.id)}" label="${esc(m.search.slice(0,115) || m.smiles.slice(0,90))}"></option>`).join("");
}
function resolve(input, allowed=null) {
  const q = input.trim().toLowerCase();
  const choices = catalog.molecules.filter(m=>!allowed || allowed.includes(m.id));
  const exact = choices.find(m => m.id === q || m.smiles.toLowerCase() === q);
  if (exact) return exact.id;
  const hits = choices.filter(m => `${m.id} ${m.smiles} ${m.search}`.toLowerCase().includes(q));
  if (q && hits.length === 1) return hits[0].id;
  throw new Error("Sélectionnez un identifiant dans les suggestions : recherche ambiguë ou inconnue.");
}
function updatePartners() {
  let a;
  try { a=resolve($("molecule-a").value); }
  catch { $("molecule-b").disabled=true;$("molecule-b").value="";$("molecules-b").innerHTML="";$("selection-b").textContent="Sélectionnez d’abord une molécule A valide.";return; }
  const ids=partnerIds(a);
  $("molecule-b").disabled=!ids.length;
  try { $("molecule-b").value=resolve($("molecule-b").value,ids); }
  catch { $("molecule-b").value=ids[0] || ""; }
  $("selection-b").textContent=`${ids.length} partenaire(s) de A présents dans le train`;
  autocomplete("","b");
}
function clearResults() {
  generation++;result=null;clearTimeout(debounce);
  clearStructures();
  for(const id of ["summary","profiles","table","chart","pharma"])$(id).replaceChildren();
  for(const id of ["csv","json","svg","png"])$(id).disabled=true;
  $("pair").innerHTML='<option value="">Sélectionnez le couple</option>';$("pair").value="";
  $("cyp-note").textContent="";$("audit").textContent="";$("limitations").replaceChildren();
  $("status").textContent="Sélection modifiée. Choisissez un partenaire puis explorez le couple.";
}
function makeRequest() {
  const a=resolve($("molecule-a").value),b=resolve($("molecule-b").value,partnerIds(a));
  if(!partnerIds(a).includes(b))throw new Error("B doit être un partenaire de A présent dans le train.");
  return {a,b,source:$("source").value,
    pair_id:$("pair").value || null,context:$("context").checked,threshold:num("threshold"),
    margin:num("margin"),min_support:num("min-support"),independence:$("independence").checked,
    filters:{search:$("search").value,category:$("category").value,minimum:num("minimum"),maximum:num("maximum"),
      delta_sign:$("delta-sign").value,sort:$("sort").value,soc:$("soc").value,pt:$("pt").value,
      minimum_n:num("minimum-n"),max_ci_width:num("ci-width")}};
}
async function compare() {
  const token = ++generation;
  try {
    const request = makeRequest();
    $("status").classList.remove("error"); $("status").textContent = "Lecture des sources locales…";
    const next = await fetchJSON("/api/compare",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(request)});
    if (token !== generation) return;
    result = next; result.request = request;
    const s = result.selection;
    $("pair").innerHTML = s.available_pairs.length ? s.available_pairs.map(id => `<option value="${esc(id)}">${esc(id)}</option>`).join("") : '<option value="">Couple absent</option>';
    $("pair").value = s.pair_id || "";
    const missing = !s.pair_id && !s.source.startsWith("imported_");
    $("status").textContent = missing ? "Ce couple est absent du train. Le contexte pharmacologique reste consultable ; aucune prédiction nouvelle n’est inventée." : `${s.pair_id || "Profil importé documenté"} · ${s.available_pairs.length} ligne(s) du couple · ${s.context ? "A/B = autres couples (pas monothérapies)" : "A/B = profils individuels documentés ou non calculables"}`;
    $("threshold-note").textContent = s.source.includes("prr") ? `Détection : PRR > ${s.threshold} ; marge ${s.margin} en unités PRR. Ce seuil n’est pas un test de significativité. Référence probabiliste interdite.` : `Détection : valeur ≥ ${s.threshold}. Marge de renforcement : ${format(s.margin*100)} points pour les probabilités comparables. Une annotation 0 n’établit pas une absence biologique.`;
    render();
  } catch (error) {
    if (token !== generation) return;
    result = null;
    clearStructures();
    $("status").classList.add("error"); $("status").textContent = error.message;
    for (const id of ["summary","profiles","table","chart"]) $(id).replaceChildren();
  }
}
function queueCompare() { clearTimeout(debounce); debounce = setTimeout(compare,220); }
function selectEffect(effect) { selected = selected === effect ? null : effect; if(result) render(); }
function bindSelections() {
  document.querySelectorAll("[data-effect]").forEach(el => {
    el.onclick = () => selectEffect(el.dataset.effect);
    if (el.tagName.toLowerCase() === "g") el.onkeydown = event => {
      if (["Enter"," "].includes(event.key)) { event.preventDefault(); selectEffect(el.dataset.effect); }
    };
  });
}
function renderPharma() {
  const p = result.pharmacology;
  $("cyp-note").textContent = p.shared_cyp_mentions.length ? `Mentions CYP communes : ${p.shared_cyp_mentions.map(c=>"CYP"+c).join(", ")}. Correspondance textuelle uniquement : aucun rôle inhibiteur, inducteur ou substrat n’est déduit.` : "Aucune mention CYP commune détectée par recherche textuelle. Cela n’exclut pas une interaction. Les mécanismes et métabolismes restent consultables ci-dessous.";
  $("pharma").innerHTML = ["a","b"].map(side => {
    const m = p[side]; $("selection-"+side).textContent = `${m.id} · nom non fourni${side==='b'?' · '+partnerIds(result.selection.a).length+' partenaire(s) de A':''}`;
    return `<article class="pharma-card"><h3>${side.toUpperCase()} · ${esc(m.id)}</h3><p>Texte source : ${esc(m.text_source)} · ${m.text_variants} variante(s). Les descriptions ne sont pas des monographies vérifiées.</p><details><summary>Structure SMILES et propriétés RDKit</summary><pre>${esc(m.smiles)}</pre><pre>${esc(JSON.stringify(m.properties,null,2))}</pre></details>${Object.entries(FIELDS).map(([key,label])=>`<details${key==="Mechanism"?' open':''}><summary>${label}</summary><p>${esc(m.fields[key] || "Non fourni").replace(/\\r\\n|\n/g,"<br>")}</p></details>`).join("")}</article>`;
  }).join("");
}
function clearStructures() {
  if($("structure-dialog").open)$("structure-dialog").close();
  $("structures-section").hidden=true;
  for(const entry of structureURLs.values())URL.revokeObjectURL(entry.url);
  structureURLs.clear();
  for(const side of ["a","b"])$("structure-img-"+side).removeAttribute("src");
  $("structure-dialog-img").removeAttribute("src");
}
function renderStructures() {
  $("structures-section").hidden=false;
  for(const side of ["a","b"]){
    const m=result.pharmacology[side],s=m.structure;
    $("structure-title-"+side).textContent=`${side.toUpperCase()} · ${m.id}`;
    const img=$("structure-img-"+side);
    img.alt=`Structure topologique 2D de ${side.toUpperCase()} : ${m.id}`;
    if(!s?.svg){img.removeAttribute("src");$("structure-meta-"+side).textContent="Structure non disponible dans cet instantané. Régénérez le HTML.";}
    else {
      const previous=structureURLs.get(side);
      if(!previous || previous.id!==m.id){
        if(previous)URL.revokeObjectURL(previous.url);
        const url=URL.createObjectURL(new Blob([s.svg],{type:"image/svg+xml;charset=utf-8"}));
        structureURLs.set(side,{id:m.id,url});
      }
      img.src=structureURLs.get(side).url;
      $("structure-meta-"+side).textContent=`${s.atoms} atomes explicites · ${s.bonds} liaisons · ${s.fragments} fragment(s) · charge totale ${s.formal_charge>0?'+':''}${s.formal_charge}. Source : SMILES du train ; RDKit ${s.rdkit_version}.`;
    }
    $("structure-enlarge-"+side).disabled=!s?.svg;$("structure-save-"+side).disabled=!s?.svg;
  }
}
function enlargeStructure(side) {
  const m=result?.pharmacology[side],entry=structureURLs.get(side);
  if(!m || !entry)return;
  $("structure-dialog-title").textContent=`${side.toUpperCase()} · ${m.id} — Structure topologique 2D`;
  $("structure-dialog-img").src=entry.url;$("structure-dialog-img").alt=`Structure agrandie de ${m.id}`;
  $("structure-dialog").showModal();
  structureBaseWidth=Math.max(640,$("structure-viewport").clientWidth-24);
  $("structure-zoom").value="100";updateStructureZoom();
}
function updateStructureZoom() {
  const value=Number($("structure-zoom").value),img=$("structure-dialog-img");
  img.width=Math.round(structureBaseWidth*value/100);img.height=Math.round(img.width*400/640);
  $("structure-zoom-value").textContent=`${value} %`;
}
function saveStructure(side) {
  const m=result?.pharmacology[side];
  if(m?.structure?.svg)download(new Blob([m.structure.svg],{type:"image/svg+xml;charset=utf-8"}),`DeepPharma_${side.toUpperCase()}_${m.id}_structure.svg`);
}
function renderSummary() {
  const s=result.summary, rows=result.rows;
  const monoKnown = rows.length && rows.every(r=>r.a.kind !== "context_frequency" && r.detected[0]!=null && r.detected[1]!=null);
  const comparisonKnown = rows.length && rows.every(r=>r.delta!=null || r.prr_deltas!=null);
  const metric = (label,value,note="Sur les effets filtrés") => `<div class="metric"><span>${esc(label)}</span><strong>${value==null?"NC":esc(format(value,2))}</strong><span>${esc(note)}</span></div>`;
  $("summary").innerHTML = metric("Détectés A",s.detected[0],result.selection.context?"Autres couples, seuil affiché":"Profils individuels") + metric("Détectés A+B",s.detected[2]) + metric("Détectés B",s.detected[1],result.selection.context?"Autres couples, seuil affiché":"Profils individuels") + metric("Jaccard A / B",s.jaccard,result.selection.context?"Similarité des contextes seulement":"Sur effets filtrés") + metric("Partagés",monoKnown?s.categories.shared||0:null) + metric("Émergents",monoKnown && rows.every(r=>r.detected[2]!=null)?s.categories.emerging||0:null) + metric("Renforcés",comparisonKnown?s.categories.strengthened||0:null) + metric("Atténués",comparisonKnown?s.categories.attenuated||0:null);
  $("population").textContent = `${rows.length} / ${result.provenance.effects} effets · seuil ${result.selection.threshold}`;
}
function renderProfiles() {
  const rows = result.rows;
  const columns=[{key:"a",index:0,title:result.selection.context?"A · autres couples":"A · seul"},{key:"ab",index:2,title:"Couple A+B"},{key:"b",index:1,title:result.selection.context?"B · autres couples":"B · seul"}];
  $("profiles").innerHTML = columns.map(c=>{
    const items=rows.filter(r=>r.detected[c.index]===true);
    const unknown=rows.filter(r=>r.detected[c.index]==null).length;
    return `<article class="profile"><h3>${c.title}</h3><p class="muted">${items.length} détecté(s) · ${unknown} non calculable(s)</p><div class="profile-items">${items.map(r=>`<button class="effect-chip${selected===r.effect?' active':''}" data-effect="${esc(r.effect)}" title="${esc(formatCell(r[c.key]))}">${esc(r.label)}</button>`).join("") || '<span class="muted">Aucun effet détecté calculable.</span>'}</div></article>`;
  }).join("");
}
function renderTable() {
  $("a-heading").textContent=result.selection.context?"A · contexte (%)":"A · seul";
  $("b-heading").textContent=result.selection.context?"B · contexte (%)":"B · seul";
  const field=c=>`<td class="${c.value==null?'nc':''}">${esc(formatCell(c))}${c.n!=null?`<small>n=${esc(c.n)} ${c.kind==='context_frequency'?'couples (pas patients)':''}${c.positives!=null?` ; k=${c.positives}`:''}</small>`:''}${c.ci?`<small>IC fourni [${format(c.ci[0])}, ${format(c.ci[1])}]</small>`:''}</td>`;
  $("table").innerHTML=result.rows.map(r=>`<tr class="${selected===r.effect?'active':''}"><td><button class="text-button" data-effect="${esc(r.effect)}">${esc(r.label)}</button><small>${esc(r.effect)}${r.pt?' · PT '+esc(r.pt):''}${r.soc?' · SOC '+esc(r.soc):''}</small></td>${field(r.a)}${field(r.b)}${field(r.ab)}<td title="${esc(r.reason)}">${r.reference==null?'Non calculable':format(r.reference*100,1)+' %'}<small>${esc(r.reason)}</small></td><td>${r.delta==null?'Non calculable':format(r.delta*100,2)+' pp'}${r.prr_deltas?`<small>Δ PRR vs A/B : ${r.prr_deltas.map(v=>format(v)).join(' / ')}<br>ln ratios : ${r.prr_log_ratios?r.prr_log_ratios.map(v=>format(v)).join(' / '):'NC (PRR nul)'}</small>`:''}</td><td>${r.categories.map(c=>`<span class="tag">${esc(CATEGORIES[c])}</span>`).join('') || 'Non classable'}</td><td>${esc(r.ab.source)}<small>n clinique ${r.ab.n??'non fourni'} · IC ${r.ab.ci?'fourni':'non fourni'}</small><small>A : ${esc(r.a.source || 'non fourni')}<br>B : ${esc(r.b.source || 'non fourni')}</small></td></tr>`).join("");
}
function svgText(x,y,text,extra="") {return `<text x="${x}" y="${y}" font-family="Segoe UI, sans-serif" font-size="13" ${extra.includes('fill=')?'':'fill="#263e52"'} ${extra}>${esc(text)}</text>`;}
function group(r,content) {return `<g tabindex="0" role="button" aria-label="${esc(r.label)}" data-effect="${esc(r.effect)}"><title>${esc(`${r.label}\nA : ${formatCell(r.a)}\nB : ${formatCell(r.b)}\nA+B : ${formatCell(r.ab)}\n${r.reason}`)}</title>${content}</g>`;}
function renderChart() {
  const rows=result.rows;
  const calculable=rows.filter(r=>r.delta!=null);
  document.querySelectorAll("[data-view]").forEach(b=>{
    b.disabled=["delta","scatter"].includes(b.dataset.view) && !calculable.length;
    b.setAttribute("aria-pressed",String(b.dataset.view===view));
    b.title=b.disabled?"Non calculable : trois probabilités comparables requises":"";
  });
  if (!calculable.length && ["delta","scatter"].includes(view)) view="bars";
  $("chart-note").textContent=result.selection.context?"A/B : proportions d’autres couples annotés (en %). A+B : annotation binaire 0/1. Échelles de nature différente : aucun delta ni conclusion d’interaction. Survolez les éléments pour les valeurs exactes." : result.selection.source.includes("prr")?"PRR bruts, échelle linéaire. Zéro reste zéro ; données manquantes sans barre. Aucune conversion en probabilité." : "Valeurs dans leur échelle d’origine ; les probabilités prédisent l’annotation du fichier, pas une incidence clinique. Les valeurs manquantes sont marquées NC.";
  if (!rows.length) {$("chart").innerHTML='<p class="chart-empty">Aucun effet ne satisfait tous les filtres. Élargissez les critères.</p>';return;}
  const W=1100, left=390, right=1040, top=85;
  let H, parts=[];
  const keys=["a","b","ab"];
  const max=Math.max(1,...rows.flatMap(r=>keys.map(k=>r[k].value??0)));
  const label=(r,y)=>svgText(left-15,y,r.label,'text-anchor="end"') + (selected===r.effect?`<circle cx="14" cy="${y-4}" r="5" fill="#2468db"/>`:'');
  if(view==="bars") {
    H=top+rows.length*96+50;
    parts.push(svgText(18,28,result.selection.context?"A / B : contexte de coadministration · AB : annotation":"Valeurs de profils · mêmes effets que le tableau"));
    keys.forEach((k,i)=>{parts.push(`<rect x="${390+i*165}" y="18" width="14" height="14" fill="${COLORS[k]}"/>`,svgText(410+i*165,30,k.toUpperCase()+(result.selection.context && k!=="ab"?" (contexte)":"")));});
    for(let t=0;t<=4;t++){const x=left+(right-left)*t/4;parts.push(`<line x1="${x}" x2="${x}" y1="55" y2="${H-35}" stroke="#d7e1eb"/>`,svgText(x,53,format(max*t/4,2),'text-anchor="middle"'));}
    rows.forEach((r,j)=>{const y=top+j*96;let elements=[label(r,y+28)];keys.forEach((k,i)=>{const v=r[k].value,yy=y+i*24;
      elements.push(v==null?svgText(left+8,yy+12,"NC"):`<rect x="${left}" y="${yy}" width="${(right-left)*v/max}" height="15" rx="3" fill="${COLORS[k]}"/>`);
      elements.push(svgText(right+12,yy+12,v==null?"":formatCell(r[k])));
    });parts.push(group(r,elements.join("")));});
  } else if(view==="heatmap") {
    H=top+rows.length*36+65;
    parts.push(svgText(18,28,`Couleur : intensité relative 0 → ${format(max)} ; gris hachuré = non calculable`));
    const xs=[440,635,830];keys.forEach((k,i)=>parts.push(svgText(xs[i]+80,57,k.toUpperCase()+(result.selection.context&&k!=="ab"?" · contexte":""),'text-anchor="middle"')));
    rows.forEach((r,j)=>{const y=top+j*36;let elements=[label(r,y+19)];keys.forEach((k,i)=>{
      const v=r[k].value;const intensity=v==null?0:v/max;const light=97-intensity*55;
      elements.push(`<rect x="${xs[i]}" y="${y}" width="165" height="29" rx="3" fill="${v==null?'url(#missing)':`hsl(211 65% ${light}%)`}" stroke="#d0dae5"/>`,svgText(xs[i]+82,y+19,v==null?"NC":formatCell(r[k]),`text-anchor="middle"${intensity>.6?' fill="#ffffff"':''}`));
    });parts.push(group(r,elements.join("")));});
  } else if(view==="delta") {
    H=top+calculable.length*40+40;const bound=Math.max(.01,...calculable.map(r=>Math.abs(r.delta)));const center=(left+right)/2,scale=(right-left)/2/bound;
    parts.push(svgText(18,28,"Écart exploratoire à l’indépendance (points de pourcentage)"));
    [-bound,0,bound].forEach(t=>parts.push(svgText(center+t*scale,57,format(t*100,1),'text-anchor="middle"')));
    parts.push(`<line x1="${center}" x2="${center}" y1="65" y2="${H-20}" stroke="#5b7388"/>`);
    calculable.forEach((r,j)=>{const y=top+j*40;parts.push(group(r,label(r,y+18)+`<rect x="${center+Math.min(0,r.delta)*scale}" y="${y}" width="${Math.abs(r.delta)*scale}" height="24" fill="${r.delta>=0?'#7146ba':'#176c86'}"/>`+svgText(right+12,y+17,format(r.delta*100,2)+" pp")));});
  } else {
    H=710;const origin=110,bottom=610,size=520;
    parts.push(svgText(18,28,"Référence indépendante vs probabilité du couple · diagonale y=x"));
    for(let t=0;t<=4;t++){const v=t/4,x=origin+size*v,y=bottom-size*v;parts.push(`<line x1="${x}" x2="${x}" y1="90" y2="${bottom}" stroke="#d7e1eb"/>`,`<line x1="${origin}" x2="${origin+size}" y1="${y}" y2="${y}" stroke="#d7e1eb"/>`,svgText(x,bottom+25,v*100+" %",'text-anchor="middle"'),svgText(origin-14,y+4,v*100+" %",'text-anchor="end"'));}
    parts.push(`<line x1="${origin}" y1="${bottom}" x2="${origin+size}" y2="${bottom-size}" stroke="#597089" stroke-dasharray="6 5"/>`,svgText(270,675,"Référence indépendante (%)"),svgText(18,65,"A+B (%)"));
    calculable.forEach(r=>parts.push(group(r,`<circle cx="${origin+size*r.reference}" cy="${bottom-size*r.ab.value}" r="${selected===r.effect?9:6}" fill="${selected===r.effect?'#a94813':'#7146ba'}" opacity=".8"/>`)));
    parts.push(svgText(710,125,"Survolez ou sélectionnez un point."),svgText(710,155,"Plus haut que y=x ≠ synergie causale."));
  }
  $("chart").innerHTML=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="${esc(view)} : ${rows.length} effets filtrés"><title>DeepPharma · ${esc(view)}</title><defs><pattern id="missing" width="7" height="7" patternUnits="userSpaceOnUse"><rect width="7" height="7" fill="#edf1f5"/><path d="M0 7L7 0" stroke="#becbd7"/></pattern></defs><rect width="100%" height="100%" fill="#fbfcfe"/>${parts.join("")}</svg>`;
}
function render() {
  renderStructures();renderPharma();renderSummary();renderProfiles();renderTable();renderChart();bindSelections();
  $("limitations").innerHTML=result.limitations.map(s=>`<li>${esc(s)}</li>`).join("");
  $("audit").textContent=JSON.stringify({selection:result.selection,filters:result.request.filters,provenance:result.provenance},null,2);
  for (const id of ["csv","json","svg","png"]) $(id).disabled=!result.rows.length;
}
function download(blob,name) {const url=URL.createObjectURL(blob);const a=document.createElement("a");a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
function csvString(data) {
  const quote=x=>{let s=String(x??"");if(typeof x!=="number" && /^[=+@\-]/.test(s))s="'"+s;return '"'+s.replace(/"/g,'""')+'"';};
  const headers=["effect","label","A","A_kind","A_n","A_source","B","B_kind","B_n","B_source","AB","AB_kind","AB_n","AB_CI","AB_source","reference","delta_pp","reason","categories","PRR_deltas","PRR_log_ratios","pair_id","train_sha256","threshold","margin","filters_json"];
  const lines=data.rows.map(r=>[r.effect,r.label,r.a.value,r.a.kind,r.a.n,r.a.source,r.b.value,r.b.kind,r.b.n,r.b.source,r.ab.value,r.ab.kind,r.ab.n,JSON.stringify(r.ab.ci),r.ab.source,r.reference,r.delta==null?null:r.delta*100,r.reason,r.categories.join("|"),JSON.stringify(r.prr_deltas),JSON.stringify(r.prr_log_ratios),data.selection.pair_id,data.provenance.train_sha256,data.selection.threshold,data.selection.margin,JSON.stringify(data.request.filters)]);
  return "\ufeff"+[headers,...lines].map(row=>row.map(quote).join(";")).join("\r\n");
}
function exportSVG() {return new XMLSerializer().serializeToString($("chart").querySelector("svg"));}
async function png() {
  if(!result)return;
  const svg=$("chart").querySelector("svg");if(!svg)return;
  const url=URL.createObjectURL(new Blob([exportSVG()],{type:"image/svg+xml"}));
  try {
    const img=new Image();img.src=url;await img.decode();
    const canvas=document.createElement("canvas");canvas.width=1100;canvas.height=Math.ceil(svg.viewBox.baseVal.height);
    canvas.getContext("2d").drawImage(img,0,0,canvas.width,canvas.height);
    const blob=await new Promise(resolve=>canvas.toBlob(resolve,"image/png"));
    if(!blob)throw new Error("Export PNG indisponible");download(blob,"deep-pharma-profils.png");
  } catch(error) {$("status").textContent=error.message;} finally {URL.revokeObjectURL(url);}
}
function clearFilters() {
  for(const id of ["search","category","minimum","maximum","delta-sign","soc","pt","ci-width","minimum-n"])$(id).value="";
  $("sort").value="name";$("min-support").value="0";queueCompare();
}
async function init() {
  try {
    catalog=await fetchJSON("/api/catalog");autocomplete();
    $("molecule-a").value=catalog.default[0];$("molecule-b").value=catalog.default[1];
    updatePartners();
    if(catalog.oof_available)$("source").insertAdjacentHTML("beforeend",'<option value="oof">Prédit · probabilité OOF (recherche binaire)</option>');
    if(catalog.audit.import_sha256)$("source").insertAdjacentHTML("beforeend",'<option value="imported_probability">Importé documenté · probabilités</option><option value="imported_prr">Importé documenté · PRR</option>');
    for(const key of ["soc","pt"]){if(catalog[key].length){$(key).disabled=false;$(key).innerHTML='<option value="">Tous</option>'+catalog[key].map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join("");}}
    // Enable clinical uncertainty filters only on documented imports, never synthesize intervals.
    $("ci-width").disabled=!catalog.audit.clinical_ci;$("minimum-n").disabled=!catalog.audit.clinical_counts;
    $("molecule-a").addEventListener("input",()=>{clearResults();autocomplete($("molecule-a").value,"a");updatePartners();});
    $("molecule-a").addEventListener("change",()=>{updatePartners();queueCompare();});
    $("molecule-b").addEventListener("input",()=>{clearResults();autocomplete($("molecule-b").value,"b");});
    $("molecule-b").addEventListener("change",queueCompare);
    $("compare").onclick=compare;
    $("swap").onclick=()=>{try{const a=resolve($("molecule-a").value),b=resolve($("molecule-b").value,partnerIds(a));$("molecule-a").value=b;$("molecule-b").value=a;updatePartners();compare();}catch(error){$("status").classList.add("error");$("status").textContent=error.message;}};
    $("source").onchange=()=>{const prr=$("source").value.includes("prr");$("threshold").value=prr?"0":"0.5";$("context").disabled=$("source").value!=="binary";$("pair").value="";queueCompare();};
    for(const id of ["pair","context","category","delta-sign","sort","soc","pt","independence"])$(id).addEventListener("change",queueCompare);
    for(const id of ["search","minimum","maximum","threshold","margin","min-support","ci-width","minimum-n"])$(id).addEventListener("input",queueCompare);
    $("reset").onclick=clearFilters;
    for(const side of ["a","b"]){$("structure-enlarge-"+side).onclick=()=>enlargeStructure(side);$("structure-save-"+side).onclick=()=>saveStructure(side);}
    $("structure-close").onclick=()=>$("structure-dialog").close();
    $("structure-zoom").addEventListener("input",updateStructureZoom);
    $("pharma-toggle").onclick=()=>{const hidden=!$("pharma").hidden;$("pharma").hidden=hidden;$("pharma-toggle").setAttribute("aria-expanded",String(!hidden));$("pharma-toggle").textContent=hidden?"Afficher les fiches":"Masquer les fiches";};
    document.querySelectorAll("[data-view]").forEach(button=>button.onclick=()=>{view=button.dataset.view;renderChart();bindSelections();});
    $("csv").onclick=()=>{if(result)download(new Blob([csvString(result)],{type:"text/csv;charset=utf-8"}),"deep-pharma-profils.csv");};
    $("json").onclick=()=>{if(result)download(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}),"deep-pharma-profils-provenance.json");};
    $("svg").onclick=()=>{if(result&&$("chart").querySelector("svg"))download(new Blob([exportSVG()],{type:"image/svg+xml"}),"deep-pharma-profils.svg");};
    $("png").onclick=png;
    await compare();
  } catch(error) {$("status").classList.add("error");$("status").textContent=`${globalThis.DeepPharmaOffline?'Chargement du HTML autonome impossible':'Service local indisponible'} : ${error.message}`;}
}
init();
