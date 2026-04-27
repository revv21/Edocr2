Let me read the current IFMEA step carefully first.
Now I also need to check the IFMEAInterface type and generateIfmeaCauses handler:
Now I also need to check how modesByFocus is structured so I know how to pull DFMEA modes per interface:
I now have the complete picture. Here are all the changes needed, explained precisely.
Changes needed
1. Update IFMEAInterface type
Remove modes, modesLoading, modesGenerated — those fields supported the old generate-your-own-modes flow. Add focusFn so each interface knows which DFMEA focus function's modes to use.
Before:
Ts
After:
Ts
You can also delete IFMEAModeRec type entirely since it's no longer needed.
2. Update the bConns sync useEffect
This is where ifmeaInterfaces is built from bConns. The interface must know which focus function it connects through. The B-diagram has lower → focus → higher connections. For a connection between a lower element and the focus, the focus function is implicit (all focus functions apply). For lower → higher connections that pass through the focus, same logic.
The simplest reliable approach: don't try to auto-detect the focus function from the connection keys. Instead store focusFn: "" and focusFnId: "" initially, and let the user select it per interface in Phase 1. This is cleaner because one B-diagram connection can be relevant to multiple focus functions.
Ts
3. Add updateIfmeaFocusFn helper
After updateIfmeaTransfer:
Ts
4. Update generateIfmeaCauses
Replace failure_modes: selectedModes (which was from iface.modes) with dfmea_failure_modes read from modesByFocus[iface.focusFnId]. Also add focus_function to the request body.
Before:
Ts
After:
Ts
5. Update buildIfmeaRows — pass interface_cause to effects endpoint
The updated effects endpoint now accepts interface_cause for a richer prompt. Add it to the row payload:
Ts
6. Replace Phase 1 UI entirely
This is the biggest visible change. Phase 1 no longer has a "Generate modes" button. Replace the entire ifmeaPhase === "modes" block with this logic:
What to show per interface card:
Header: same as before (connection type badge, from → to, connection label)
Nominal transfer input: same as before
Focus function selector: a <select> or set of radio pills showing all focus functions. The user picks which focus function this interface relates to.
DFMEA modes display: once a focus function is selected, show the modes already selected in Step 6 for that function as a read-only list with tick icons. These are the modes that will be used for cause generation — no generate button, no checkboxes.
Warning if focus function has no selected modes yet: "No failure modes selected for this function yet — go to Step 6 first."
Tsx
7. Update generateIfmeaCauses guard condition
The old guard was !selectedModes.length || !iface.nominalTransfer.trim(). The new guard is:
Ts
8. Delete fetchIfmeaModes handler
The entire fetchIfmeaModes async function can be deleted — it calls /api/ifmea/interface-failure-modes which no longer exists in the backend.
9. Update the Phase tab label
Change "1. Failure Modes" to "1. Set Up Interfaces" in the phase tabs array to reflect that this phase now configures rather than generates:
Ts
Summary of what gets removed vs added
Removed:
modes, modesLoading, modesGenerated fields on IFMEAInterface
IFMEAModeRec type
fetchIfmeaModes async function (~20 lines)
"Generate modes" button per interface card
The iface.modes.filter(m => m.selected) logic in generateIfmeaCauses
Added:
focusFn and focusFnId fields on IFMEAInterface
updateIfmeaFocusFn helper (~6 lines)
Focus function pill selector per interface card
Read-only DFMEA modes display per interface card
dfmea_failure_modes and focus_function in the cause generation request body
interface_cause in the effects request body