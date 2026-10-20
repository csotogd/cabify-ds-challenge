"""Notebook cells for the explicitly exploratory road extension."""
import nbformat as nbf


def setup_cells():
    return [nbf.v4.new_markdown_cell(r'''
## 4. From the drawing of a route to the streets it can actually use

### 4.1 An API failed; the hypothesis was still testable

The first attempt with Overpass only got a small area of Madrid, with no complete journeys. That was a map access problem, not evidence against graphs. We now download full extracts for eight countries from [Geofabrik](https://download.geofabrik.de/), with **OpenStreetMap** data. We record the date and fingerprint of each file, and clip an area around the traces with a margin of 0.08 degrees per side. The margin makes it easier to drive around the trip, but does not guarantee every possible detour is included.

**A current map does not reconstruct a historical one.** Streets, accesses or directions may have changed since these trips. We also have no timestamps and no observed traffic. That is why the duration variables we add are estimates under a fixed car profile, not the real time each vehicle took.

### 4.2 First we need to know which streets each route used

A noisy coordinate can land between two streets. Assigning each point to the nearest street independently can produce impossible jumps. We use the *map matching* service of [OSRM](https://project-osrm.org/docs/v5.24.0/api/#match-service): it looks for a road sequence compatible with the points and returns segments, rejected points and a confidence measure.

The engine runs locally with a pinned version and the `car.lua` profile, which encodes directions and the turn restrictions that profile supports and the map contains. **We do not replace each journey with the fastest route between its origin and destination:** that would erase precisely the detours we want to detect.

To limit computation we keep up to 300 **observed** points, chosen by cumulative distance, endpoints included. We invent no intermediate observations. We assume a GPS accuracy of 15 metres to generate candidates; the file does not provide that accuracy. This choice can influence the result and deserves a sensitivity analysis. Without timestamps, we impose no observed speeds between points.

OSRM's confidence is not a calibrated probability for this dataset. We call a route a «complete match under our rule» when it has a single sub-trace, at least 90 % of selected points accepted and confidence ≥ 0.5. It is an explicit diagnostic rule, not a guarantee of truth. **No journey is removed from the benchmark for failing it**: we keep quality indicators and missing values where appropriate.

### 4.3 What we measure on the graph and what each number means

A directed edge represents a street segment in one direction. That lets us tell sharing a street apart from driving it in opposite directions. We build two related representations:

| Representation | What it is used for | Relevant limitation |
|---|---|---|
| OSRM, with the car profile | Infer the trace and estimate the length and duration of its segments | Current map, fixed profile; no traffic and no historical duration |
| Directed graph of OSM nodes, weighted in metres | Minimum distance from the nodes of one trace to those of the other | Respects simplified static directions and accesses; **does not model turn restrictions**, conditional accesses or ferries |

For a route A and a route B, on this second graph we define

$$d_G(u,B)=\min_{v\in V_B}\operatorname{dist}_G(u,v).$$

The distance follows the arrows. That is why $d_G(u,B)$ and $d_G(v,A)$ need not coincide. We compute both directions with multi-target Dijkstra, run over the reversed graph. The target is **any node of the other trace**; we do not require it to be at the same stage of the trip.

This is a node-level approximation, not the exact distance between all continuous points of two curves. We average with length weights: each endpoint node is assigned half the metres actually driven on that segment according to OSRM. A stretch split by several GPS points does not count its full length again at each split. On partial segments we approximate the position using the nodes of the full edge; that can introduce error near endpoints or on long edges.

We store the mean, 95th percentile, maximum and proportion of length beyond 50, 100 and 500 metres, in both directions. To limit search cost, we truncate distances at 10 km and add the fraction that exceeds that limit or is unreachable. **With truncation, the mean and the maximum are lower bounds**, not exact distances. A clip of the extract can also cause unreachability.

Street overlap uses unique directed edges and their lengths:

$$\operatorname{IoU}_{\mathrm{streets}}(A,B)=
\frac{\sum_e\min(w_A(e),w_B(e))}{\sum_e\max(w_A(e),w_B(e))}.$$

Here $w_A(e)$ is the edge length if it appears in A and zero otherwise. A repeated edge counts once. Touching part of an edge counts as touching all of it: this approximation can overestimate overlap if the routes use different portions of the same edge. This IoU also does not represent global ordering or how many times a street is driven. It complements the ordering and geometry measures we already have.

Finally we keep estimated lengths and durations of the **matched portions** and their quality indicators. We only compute relative length and time differences as comparable totals when both routes satisfy the match rule. For instance:

$$\Delta T_{\mathrm{rel}}=\frac{\hat T_r-\hat T_e}{\max[(\hat T_r+\hat T_e)/2,1\text{ s}]}.$$

The division compares the relative overrun; we also keep the individual values. IoU and the proportions are already normalised. **The label is still the human judgement about similarity**, not «it takes longer»: the model has to learn whether these variables help predict that judgement.

### 4.4 Coverage is a result that has to be shown

This extension was proposed after inspecting the original test set. A new protocol was frozen before obtaining its results and the same journeys were kept. Its test metrics are **exploratory**, not an independent confirmation of an improvement.
'''),nbf.v4.new_code_cell(r'''
RG=ROOT/'results/routes_graph'
def read_graph(name):return json.loads((RG/name).read_text())
graph_manifest=read_graph('feature_manifest.json')
extract_manifest=read_graph('extract_manifest.json')
node_manifest=read_graph('node_graph.json')
graph_metrics=read_graph('metrics.json')
graph_frozen=read_graph('frozen_selection.json')
graph_details=read_graph('training_details.json')
graph_features=pd.read_json(ROOT/'data/cache/roads/graph_features.json',orient='index').loc[df.index]
graph_predictions=pd.DataFrame(read_graph('predictions.json')).set_index('journey_id').loc[df.index]
display(Markdown(f'**Map imported:** {extract_manifest["nodes"]:,} OSM nodes; '
    f'**simplified drivable graph:** {node_manifest["nodes"]:,} nodes and {node_manifest["directed_edges"]:,} edges. '
    f'We build **{graph_manifest["n_features"]} road and quality variables** per journey.'))
coverage=pd.DataFrame(graph_manifest['coverage']).set_index('split')
display(coverage[['n','both_any_match','both_reliable']].rename(columns={
    'n':'Journeys','both_any_match':'Some match on both','both_reliable':'Both satisfy the rule'}))
fig,ax=plt.subplots(figsize=(10,4))
x=np.arange(3);n=coverage.n.to_numpy();good=coverage.both_reliable.to_numpy();anymatch=coverage.both_any_match.to_numpy()
ax.bar(x,good,color=TEAL,label='Both satisfy the rule')
ax.bar(x,anymatch-good,bottom=good,color=ORANGE,label='Match on both, with reservations')
ax.bar(x,n-anymatch,bottom=anymatch,color=RED,label='Match missing on at least one')
for i in x:ax.text(i,n[i]+15,f'{good[i]}/{n[i]} complete',ha='center',fontsize=10)
ax.set_xticks(x,['Training','Validation','Test']);ax.set_ylabel('Journeys');ax.set_ylim(0,n.max()*1.2)
ax.set_title('Downloading the map does not guarantee reconstructing every journey');ax.legend(fontsize=9,loc='upper right')
finish(fig,'04_cobertura_grafo','Pre-set diagnostic rule · no journey is dropped from the metrics for match quality')
''')]


def result_cells():
    return [nbf.v4.new_markdown_cell(r'''
### 7.1 Exploratory extension: now we can actually test the road network

We compare two new variants: **graph + XGBoost** and **42 geometric measurements + graph + XGBoost**. We use the same three XGBoost configurations and the same selection rule by validation log loss as in the original experiment. Thresholds are chosen on validation by balanced accuracy. We do not retrain on test and we do not discard journeys that are hard to match.

Before evaluating test we record which of the two variants wins on validation. The original geometric baseline and its predictions stay untouched. The test set had already guided the earlier discussion: keeping its labels out of fitting these models avoids a direct leak, but **it does not turn a comparison made after inspecting it into a fresh confirmatory test**.
'''),nbf.v4.new_code_cell(r'''
new_names={'grafo_xgb':'Graph','geometria_grafo_xgb':'Geometry + graph'}
rows=[]
for name in ['geometria_base','vector_interpretable']:
    m=metrics[name]
    rows.append({'Model':names[name],'Variables':8 if name=='geometria_base' else 42,
       'LL val ↓':details[name]['val_soft_logloss'],'BA val ↑':m['validation']['balanced_accuracy'],
       'LL test ↓':m['soft_logloss_all_decisive_votes'],'BA test ↑':m['balanced_accuracy'],
       'F1 test ↑':m['f1'],'AUC test ↑':m['roc_auc'],'Threshold':m['threshold']})
for name,m in graph_metrics.items():
    rows.append({'Model':new_names[name],'Variables':graph_details[name]['n_features'],
       'LL val ↓':m['val_soft_logloss'],'BA val ↑':m['validation']['balanced_accuracy'],
       'LL test ↓':m['test_soft_logloss'],'BA test ↑':m['balanced_accuracy'],
       'F1 test ↑':m['f1'],'AUC test ↑':m['roc_auc'],'Threshold':m['threshold']})
road_table=pd.DataFrame(rows).set_index('Model')
display(road_table.style.format({c:'{:.4f}' for c in road_table.columns if c!='Variables'}))
selected_graph=graph_frozen['selected_new_variant'];gm=graph_metrics[selected_graph]
ci=gm['delta_base_ci95'];delta=100*gm['delta_base']
comparison='includes zero: it does not demonstrate a stable improvement' if ci[0]<=0<=ci[1] else ('lies above zero' if ci[0]>0 else 'lies below zero')
display(Markdown(f'The new variant chosen on validation is **{new_names[selected_graph]}**. '
    f'On the same **{gm["n"]} test journeys with a majority**, the balanced accuracy difference against the baseline is '
    f'**{delta:+.2f} percentage points**, with a 95 % paired bootstrap interval '
    f'**[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]**. The interval {comparison}. '
    'It describes variability across these journeys; it does not include repeating training, selection or new cities.'))
fig,axs=plt.subplots(1,2,figsize=(12,4.8))
order=list(road_table.index);yy=np.arange(len(order));colors=[INK,TEAL,ORANGE,RED]
axs[0].barh(yy,road_table['LL val ↓'],color=colors);axs[0].set_yticks(yy,order)
axs[0].invert_yaxis();axs[0].set_title('First we select on validation');axs[0].set_xlabel('Log loss against votes · lower is better')
for i,v in enumerate(road_table['LL val ↓']):axs[0].text(v+.002,i,f'{v:.3f}',va='center',fontsize=9)
axs[0].set_xlim(0,road_table['LL val ↓'].max()*1.22)
for i,(name,m) in enumerate(graph_metrics.items()):
    lo,hi=np.array(m['delta_base_ci95'])*100;d=m['delta_base']*100
    axs[1].plot([lo,hi],[i,i],color=ORANGE if i==0 else RED,lw=4)
    axs[1].scatter([d],[i],s=65,color=INK,zorder=3)
axs[1].axvline(0,color=MUTED,ls='--');axs[1].set_yticks([0,1],[new_names[k] for k in graph_metrics])
axs[1].set_ylim(-.6,1.6);axs[1].set_xlabel('Change in balanced accuracy · percentage points')
axs[1].set_title('Then we quantify the difference on test')
finish(fig,'07b_comparacion_grafo','Exploratory extension made after inspecting the test set · bootstrap CI by journey, not by city')

# Quality against performance, without changing filters, model or threshold.
quality_rows=[]
test_ids=df.index[(df.split=='test')&df.target_hard.notna()]
for complete,label in [(1,'Both satisfy the rule'),(0,'Partial or doubtful match')]:
    ids=test_ids[graph_features.loc[test_ids,'both_reliable'].to_numpy()==complete]
    y=df.loc[ids,'target_hard'].to_numpy();p=graph_predictions.loc[ids,'probabilities'].map(lambda v:v[selected_graph]).to_numpy()
    pred=p>=gm['threshold']
    quality_rows.append({'Quality':label,'Journeys':len(ids),'Errors':int((pred!=y).sum()),
        'Accuracy':float((pred==y).mean()) if len(ids) else np.nan})
display(pd.DataFrame(quality_rows).set_index('Quality').style.format({'Accuracy':'{:.1%}'}))
'''),nbf.v4.new_markdown_cell(r'''
#### What the map lets us see that a straight-line distance cannot

We show a **training** example, chosen deterministically by match quality and central length, to inspect the transformation. The thin lines are the observations; the thick lines, the OSRM reconstruction. Looking good does not validate every journey, nor does it turn the current map into historical truth.
'''),nbf.v4.new_code_cell(r'''
eligible=df.index[(df.split=='train')&(graph_features.both_reliable==1)]
if len(eligible):
    order_ids=features.loc[eligible,'length_real'].sort_values().index
    road_example=order_ids[len(order_ids)//2]
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for ax,kind,label,route,color in zip(axs,['estimated','real'],['Estimated','Real'],routes[road_example],[INK,ORANGE]):
        raw=json.loads((ROOT/'data/cache/roads/matches'/f'{road_example}_{kind}.json').read_text())
        ax.plot(route[:,1],route[:,0],color=MUTED,alpha=.6,lw=3,label='GPS observations')
        for idx,m in enumerate(raw.get('matchings',[])):
            xy=np.asarray(m['geometry']['coordinates'])
            ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5,label='Street match' if idx==0 else None)
        ax.set_aspect(1/np.cos(np.radians(route[:,0].mean())))
        ax.set_title(label+' · GPS and road network');ax.set_xlabel('Longitude');ax.set_ylabel('Latitude');ax.legend(fontsize=9)
    finish(fig,'07c_map_matching_ejemplo','Training journey · © OpenStreetMap contributors · ODbL · the match is an inference')
    display(Markdown(f'Journey `{road_example}` · directed-edge IoU: '
        f'**{graph_features.loc[road_example,"directed_edge_iou"]:.3f}**. '
        'The durations correspond to the inferred streets under the engine\'s fixed profile.'))
'''),nbf.v4.new_markdown_cell(r'''
**What remains to improve:** manually validate a sample of matches; vary the assumed GPS accuracy and the point density; use cartography from the date of the trip; obtain timestamps; measure distances on a network with turn restrictions for the proximity variables too; and separate the effect of connectivity, times and quality through ablations. Node-level proximity should be contrasted with an edge-level approximation. We also need to evaluate an unseen city and a fresh test set if this line is to be developed further.

The quality indicators may help the classifier, but they can also capture differences between cities or devices. So an improvement of the combined model **would not by itself demonstrate that connectivity is the mechanism responsible**. The map makes the experiment possible; its predictive usefulness and its reliability remain empirical questions.
''')]
