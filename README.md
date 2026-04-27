// Select/deselect all causes across every interface group
const toggleIfmeaSelectAll = () => {
  const next = !ifmeaSelectAll;
  setIfmeaSelectAll(next);
  setIfmeaCauseGroups(p =>
    p.map(g => ({ ...g, causes: g.causes.map(c => ({ ...c, selected: next })) }))
  );
};

// Select/deselect all causes within one group
const toggleIfmeaGroupSelectAll = (gi: number) => {
  setIfmeaCauseGroups(p => {
    const group = p[gi];
    const allSelected = group.causes.every(c => c.selected);
    return p.map((g, i) =>
      i !== gi ? g : { ...g, causes: g.causes.map(c => ({ ...c, selected: !allSelected })) }
    );
  });
};{ifmeaCauseGroups.map((group, gi) => {
  const sel = group.causes.filter(c => c.selected).length;
  const allSel = group.causes.length > 0 && group.causes.every(c => c.selected);
  return (
    <Collapsible key={gi}
      title={`${group.fromElement} → ${group.toElement}  |  ${group.failureMode}`}
      badge={`${sel} / ${group.causes.length}`}
      badgeVariant={sel > 0 ? "default" : "secondary"}>
      <div className="space-y-2">
        {/* Per-group select all */}
        <div className="flex justify-end">
          <button type="button"
            className="text-xs text-primary hover:underline"
            onClick={() => toggleIfmeaGroupSelectAll(gi)}>
            {allSel ? "Deselect all" : "Select all in this group"}
          </button>
        </div>
        {!group.causes.length
          ? <p className="text-sm text-muted-foreground">No causes generated.</p>
          : group.causes.map(cause => (
              // ... existing cause label cards unchanged
            ))
        }
      </div>
    </Collapsible>
  );
})}<div className="flex items-center justify-between flex-wrap gap-2">
  <div className="flex items-center gap-2 text-sm">
    <Badge variant="outline">{ifmeaTotalSelected} selected</Badge>
    <span className="text-muted-foreground">across {ifmeaCauseGroups.length} groups</span>
  </div>
  <div className="flex items-center gap-2">
    <Button size="sm" variant="outline" onClick={toggleIfmeaSelectAll}>
      {ifmeaSelectAll ? "Deselect all" : "Select all"}
    </Button>
    <Button size="sm" variant="outline" disabled={ifmeaCausesLoading} onClick={generateIfmeaCauses}>
      {ifmeaCausesLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Regenerate"}
    </Button>
  </div>
</div>
const [ifmeaAutoRatingLoading, setIfmeaAutoRatingLoading] = useState(false);const autoAssignIfmeaRatings = async () => {
  setIfmeaAutoRatingLoading(true);

  const payload = ifmeaCauseGroups.flatMap(group =>
    group.causes
      .filter(c => c.selected)
      .map(cause => ({
        cause_id:           cause.id,
        cause:              cause.cause,
        noise_factor:       cause.noise_factor,
        noise_category:     cause.noise_category,
        failure_mode:       group.failureMode,
        focus_function:     group.failureMode,  // interface context
        focus_element:      group.fromElement,
        lower_element:      group.fromElement,  // "from" side of the interface
        lower_function:     group.nominalTransfer, // what the interface transfers
        prevention_methods: cause.prevention_methods,
        detection_methods:  cause.detection_methods,
      }))
  );

  try {
    const res = await fetch(`${apiBase}/api/dfmea/auto-rating/bulk`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ causes: payload }),
    });
    const data = await res.json();

    const resultMap: Record<string, { occurrence_answer: string; detection_answer: string }> = {};
    for (const r of data.results ?? []) resultMap[r.cause_id] = r;

    setIfmeaCauseGroups(prev =>
      prev.map(group => ({
        ...group,
        causes: group.causes.map(cause => {
          const r = resultMap[cause.id];
          if (!r) return cause;
          return { ...cause, occurrence_answer: r.occurrence_answer, detection_answer: r.detection_answer };
        }),
      }))
    );
  } catch (e) {
    console.error("IFMEA auto-rating failed:", e);
  } finally {
    setIfmeaAutoRatingLoading(false);
  }
};<div className="flex items-center gap-3 p-3 rounded-lg bg-muted/30 border">
  <div className="flex-1 h-1.5 bg-border rounded-full overflow-hidden">
    <div className="h-full bg-primary rounded-full transition-all"
      style={{ width: ifmeaTotalSelected
        ? `${(ifmeaTotalRated / ifmeaTotalSelected) * 100}%` : "0%" }} />
  </div>
  <span className="text-sm font-medium shrink-0">
    {ifmeaTotalRated} / {ifmeaTotalSelected} rated
  </span>
  {ifmeaTotalRated < ifmeaTotalSelected && (
    <span className="flex items-center gap-1 text-amber-600 text-xs shrink-0">
      <AlertTriangle className="h-3 w-3" />Incomplete
    </span>
  )}
  <Button size="sm" variant="outline"
    disabled={ifmeaAutoRatingLoading || ifmeaTotalSelected === 0}
    onClick={autoAssignIfmeaRatings}>
    {ifmeaAutoRatingLoading
      ? <><Loader2 className="h-3.5 w-3.5 mr-1.5 animate-spin" />Auto-assigning…</>
      : <><Zap className="h-3.5 w-3.5 mr-1.5" />Auto-assign risk rating</>}
  </Button>
</div>