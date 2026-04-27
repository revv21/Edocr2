7const updateIfmeaFocusFn = (id: string, focusFnId: string) => {
  const ff = focusFunctions.find(f => f.id === focusFnId);
  setIfmeaInterfaces(p => p.map(i =>
    i.id !== id ? i : { ...i, focusFnId, focusFn: ff?.name ?? "" }
  ));
};// Get the DFMEA failure modes for this interface's focus function
const dfmeaModes = Array.from(modesByFocus[iface.focusFnId]?.selected ?? []);
if (!dfmeaModes.length || !iface.nominalTransfer.trim() || !iface.focusFnId) continue;
// ...
body: JSON.stringify({
  from_element:        iface.fromElement,
  to_element:          iface.toElement,
  connection_type:     iface.connType,
  nominal_transfer:    iface.nominalTransfer,
  focus_function:      iface.focusFn,
  dfmea_failure_modes: dfmeaModes,       // renamed field, new backend expects this
  noise_factors:       cleanNoise,
}),body: JSON.stringify({
  rows: draft.map(row => ({
    row_id:           row.id,
    from_element:     row.from_element,
    to_element:       row.to_element,
    connection_type:  row.conn_type,
    nominal_transfer: row.nominal_transfer,
    failure_mode:     row.failure_mode,
    interface_cause:  row.failure_cause,  // add this line
  })),
}),{ifmeaPhase === "modes" && (
  <div className="space-y-4">
    <p className="text-xs text-muted-foreground">
      For each interface, describe what is transferred and select which focus
      function it relates to. The DFMEA failure modes for that function will
      be used as the interface failure modes — no separate generation needed.
    </p>
    {ifmeaInterfaces.map(iface => {
      const meta = CONN_META[iface.connType];
      const dfmeaModes = Array.from(modesByFocus[iface.focusFnId]?.selected ?? []);
      return (
        <Card key={iface.id} className="border">
          <CardContent className="p-4 space-y-3">
            {/* Interface header */}
            <div className="flex items-center gap-3 flex-wrap">
              <span className="px-2 py-0.5 rounded text-xs font-bold text-white"
                style={{ background: meta.color }}>{iface.connType}</span>
              <span className="font-semibold text-sm">{iface.fromElement}</span>
              <ArrowRight className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
              <span className="font-semibold text-sm">{iface.toElement}</span>
              <span className="text-xs text-muted-foreground">({meta.label})</span>
              {dfmeaModes.length > 0 && (
                <Badge variant="default" className="ml-auto">
                  {dfmeaModes.length} DFMEA modes
                </Badge>
              )}
            </div>

            {/* Nominal transfer */}
            <div className="space-y-1">
              <Label className="text-xs font-semibold">
                What is nominally transferred through this interface?
              </Label>
              <Input className="text-sm"
                placeholder={...} // same placeholders as before
                value={iface.nominalTransfer}
                onChange={e => updateIfmeaTransfer(iface.id, e.target.value)} />
            </div>

            {/* Focus function selector */}
            <div className="space-y-1.5">
              <Label className="text-xs font-semibold">
                Which focus function does this interface support?
              </Label>
              <div className="flex flex-wrap gap-2">
                {focusFunctions.map(ff => (
                  <button key={ff.id} type="button"
                    onClick={() => updateIfmeaFocusFn(iface.id, ff.id)}
                    className={`text-xs px-3 py-1.5 rounded-lg border transition-all ${
                      iface.focusFnId === ff.id
                        ? "bg-primary text-primary-foreground border-primary"
                        : "bg-white text-gray-600 border-gray-300 hover:border-primary/40"
                    }`}>
                    {ff.name}
                  </button>
                ))}
              </div>
            </div>

            {/* DFMEA modes — read only, derived from modesByFocus */}
            {iface.focusFnId && (
              <div className="space-y-1.5">
                <Label className="text-xs font-semibold text-muted-foreground">
                  DFMEA failure modes for this interface (from Step 6)
                </Label>
                {dfmeaModes.length === 0 ? (
                  <p className="text-xs text-amber-600 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2">
                    No failure modes selected for this function yet. Go to Step 6 and generate modes first.
                  </p>
                ) : (
                  <div className="space-y-1">
                    {dfmeaModes.map((mode, mi) => (
                      <div key={mi}
                        className="flex items-start gap-2 px-3 py-2 rounded-lg bg-primary/5 border border-primary/20 text-sm">
                        <CheckCircle2 className="h-3.5 w-3.5 text-primary mt-0.5 shrink-0" />
                        <span className="text-gray-700">{mode}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      );
    })}
    <div className="flex justify-end">
      <Button onClick={() => setIfmeaPhase("causes")}>
        Next: Generate Causes <ArrowRight className="h-4 w-4 ml-2" />
      </Button>
    </div>
  </div>
)}if (!iface.focusFnId || !iface.nominalTransfer.trim()) continue;
const dfmeaModes = Array.from(modesByFocus[iface.focusFnId]?.selected ?? []);
if (!dfmeaModes.length) continue;
const updateIfmeaFocusFn = (id: string, focusFnId: string) => {
  const ff = focusFunctions.find(f => f.id === focusFnId);
  setIfmeaInterfaces(p => p.map(i =>
    i.id !== id ? i : { ...i, focusFnId, focusFn: ff?.name ?? "" }
  ));
};