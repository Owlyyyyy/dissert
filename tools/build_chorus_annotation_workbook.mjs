import fs from "node:fs/promises";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const [csvPath, outputPath, previewDir] = process.argv.slice(2);
if (!csvPath || !outputPath || !previewDir) {
  throw new Error("Usage: node build_chorus_annotation_workbook.mjs input.csv output.xlsx preview_dir");
}

const source = await Workbook.fromCSV(await fs.readFile(csvPath, "utf8"), { sheetName: "Raw" });
const rawValues = source.worksheets.getItem("Raw").getUsedRange(true).values;
const headers = rawValues[0].map(String);
const records = rawValues.slice(1).map((row) => Object.fromEntries(headers.map((header, i) => [header, row[i] ?? ""])));

const workbook = Workbook.create();
const protocol = workbook.worksheets.add("Protocol");
const annotations = workbook.worksheets.add("Annotations");
const baseline = workbook.worksheets.add("Detector Baseline");

protocol.showGridLines = false;
protocol.getRange("A1:F1").merge();
protocol.getRange("A1").values = [["Chorus Annotation Protocol"]];
protocol.getRange("A1:F1").format = {
  fill: "#183153", font: { bold: true, color: "#FFFFFF" },
  verticalAlignment: "center",
};
protocol.getRange("A1:F1").format.rowHeight = 34;
const protocolRows = [
  ["Purpose", "Create human ground truth before tuning the rebuilt detector."],
  ["Blind review", "Annotate from listening. Do not consult the Detector Baseline sheet until all human fields are complete."],
  ["Chorus", "The principal repeated section or hook. Enter every audible interval as start-end; separate multiple intervals with semicolons."],
  ["Artist + Chorus headers", "[Artist: Chorus] and [Chorus: Artist] both count as chorus."],
  ["Pre-Chorus", "Record separately. It does not contribute to chorus duration."],
  ["Post-Chorus", "Record separately. It does not contribute to chorus duration."],
  ["Hook / Refrain", "Count as chorus for the primary analysis, but mention ambiguous cases in Notes."],
  ["Presence", "Use yes, no, or uncertain. Use uncertain when musical structure or boundaries cannot be judged confidently."],
  ["Interval format", "Examples: 8.40-21.70 or 2.10-8.30;18.20-29.90. Use seconds from the beginning of the preview."],
  ["Metadata", "Track-level artist is preferred. Compilation credits such as Various are not accepted when a track artist is available."],
  ["Version policy", "Keep qualifiers such as Remix, Live, Demo, Instrumental, and Remaster. Note any apparent audio/lyric version mismatch."],
  ["Reviewer", "Enter initials or a stable reviewer ID. A second reviewer should independently label at least 10 previews."],
];
protocol.getRange(`A3:B${protocolRows.length + 2}`).values = protocolRows;
protocol.getRange(`A3:A${protocolRows.length + 2}`).format = { font: { bold: true, color: "#183153" }, fill: "#EAF0F7" };
protocol.getRange(`A3:B${protocolRows.length + 2}`).format.wrapText = true;
protocol.getRange(`A1:A${protocolRows.length + 2}`).format.columnWidth = 22;
protocol.getRange(`B1:B${protocolRows.length + 2}`).format.columnWidth = 92;
protocol.getRange(`A3:B${protocolRows.length + 2}`).format.autofitRows();

const annotationHeaders = [
  "Sample ID", "Audio path", "Artist", "Title", "Metadata status", "Chorus present",
  "Chorus intervals (s)", "Pre-chorus intervals (s)", "Post-chorus intervals (s)",
  "Reviewer", "Notes",
];
const annotationRows = records.map((record) => [
  record.sample_id, record.audio_path, record.artist, record.title,
  record.metadata_status, record.human_chorus_present, record.human_chorus_intervals,
  record.human_pre_chorus_intervals, record.human_post_chorus_intervals,
  record.reviewer, record.notes,
]);
annotations.getRangeByIndexes(0, 0, 1, annotationHeaders.length).values = [annotationHeaders];
annotations.getRangeByIndexes(1, 0, annotationRows.length, annotationHeaders.length).values = annotationRows;
annotations.getRange(`A1:K${annotationRows.length + 1}`).format.wrapText = true;
annotations.getRange("A1:K1").format = { fill: "#183153", font: { bold: true, color: "#FFFFFF" }, verticalAlignment: "center" };
annotations.getRange("A1:K1").format.rowHeight = 30;
annotations.freezePanes.freezeRows(1);
annotations.freezePanes.freezeColumns(4);
annotations.showGridLines = false;
annotations.tables.add(`A1:K${annotationRows.length + 1}`, true, "ChorusAnnotations").style = "TableStyleMedium2";
annotations.getRange(`F2:F${annotationRows.length + 1}`).dataValidation = { rule: { type: "list", values: ["yes", "no", "uncertain"] } };
const widths = [12, 54, 23, 30, 20, 18, 24, 26, 27, 14, 42];
widths.forEach((width, index) => { annotations.getRangeByIndexes(0, index, annotationRows.length + 1, 1).format.columnWidth = width; });
annotations.getRange(`A2:K${annotationRows.length + 1}`).format.rowHeight = 30;

const baselineHeaders = ["Sample ID", "Artist", "Title", "Old detector status", "Old chorus %", "Old intervals", "Metadata source"];
const baselineRows = records.map((record) => [
  record.sample_id, record.artist, record.title, record.detector_status,
  record.predicted_chorus_percent === "" ? null : Number(record.predicted_chorus_percent),
  record.predicted_intervals, record.metadata_source,
]);
baseline.getRangeByIndexes(0, 0, 1, baselineHeaders.length).values = [baselineHeaders];
baseline.getRangeByIndexes(1, 0, baselineRows.length, baselineHeaders.length).values = baselineRows;
baseline.getRange("A1:G1").format = { fill: "#6B3E26", font: { bold: true, color: "#FFFFFF" } };
baseline.getRange(`A1:G${baselineRows.length + 1}`).format.wrapText = true;
baseline.getRange(`E2:E${baselineRows.length + 1}`).format.numberFormat = "0.00";
baseline.tables.add(`A1:G${baselineRows.length + 1}`, true, "DetectorBaseline").style = "TableStyleMedium9";
baseline.freezePanes.freezeRows(1);
baseline.showGridLines = false;
[12, 23, 30, 20, 16, 28, 18].forEach((width, index) => { baseline.getRangeByIndexes(0, index, baselineRows.length + 1, 1).format.columnWidth = width; });

await fs.mkdir(previewDir, { recursive: true });
for (const sheetName of ["Protocol", "Annotations", "Detector Baseline"]) {
  const image = await workbook.render({ sheetName, autoCrop: "all", scale: 1, format: "png" });
  await fs.writeFile(`${previewDir}/${sheetName.replaceAll(" ", "_")}.png`, new Uint8Array(await image.arrayBuffer()));
}
const inspection = await workbook.inspect({ kind: "table", range: "Annotations!A1:K8", include: "values,formulas", tableMaxRows: 8, tableMaxCols: 11, maxChars: 5000 });
console.log(inspection.ndjson);
const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A", options: { useRegex: true, maxResults: 100 }, summary: "formula error scan" });
console.log(errors.ndjson);
await fs.mkdir(outputPath.replace(/[\\/][^\\/]+$/, ""), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
