#!/usr/bin/env node
/**
 * Build the EdgeTwinCal lab-return workbook and machine-readable CSV tables.
 *
 * Workbook authoring deliberately uses @oai/artifact-tool. The explicit
 * repository-root option lets this file run from a temporary directory whose
 * node_modules junction points at Codex's bundled dependency runtime.
 */

import crypto from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { fileURLToPath } from 'node:url';

const CAMPAIGN = 'edgetwincal_msn2026_v1';
const REQUIRED_INPUTS = [
  'confirmatory_aggregate.json',
  'gate_decision.json',
  'failure_diagnosis.json',
];
const VARIANT_ORDER = [
  'APN', 'SLRH', 'CFG', 'Full',
  'V01', 'V02', 'V03', 'V07', 'V08', 'V10', 'V11', 'V12',
];
const SHEET_NAMES = [
  'Results',
  'Seed metrics',
  'Paired CIs',
  'Gate audit',
  'Provenance',
];
const COLORS = {
  navy: '#17324D',
  teal: '#0F766E',
  paleBlue: '#EAF2F8',
  paleGreen: '#DCFCE7',
  paleRed: '#FEE2E2',
  paleAmber: '#FEF3C7',
  paleGray: '#F8FAFC',
  white: '#FFFFFF',
  line: '#CBD5E1',
  text: '#0F172A',
};

function parseArgs(argv) {
  const scriptDir = path.dirname(fileURLToPath(import.meta.url));
  const args = {
    repoRoot: path.resolve(scriptDir, '..', '..'),
    outputDir: null,
    qaDir: null,
  };
  for (let index = 2; index < argv.length; index += 1) {
    const key = argv[index];
    const value = argv[index + 1];
    if (key === '--repo-root' && value) {
      args.repoRoot = path.resolve(value);
      index += 1;
    } else if (key === '--output-dir' && value) {
      args.outputDir = path.resolve(value);
      index += 1;
    } else if (key === '--qa-dir' && value) {
      args.qaDir = path.resolve(value);
      index += 1;
    } else if (key === '--help') {
      console.log(
        'Usage: node build_edgetwincal_tables.mjs ' +
        '[--repo-root PATH] [--output-dir PATH] [--qa-dir PATH]',
      );
      process.exit(0);
    } else {
      throw new Error('Unknown or incomplete argument: ' + key);
    }
  }
  args.analysisDir = path.join(
    args.repoRoot,
    'artifacts',
    CAMPAIGN,
    'analysis',
  );
  if (!args.outputDir) {
    args.outputDir = args.analysisDir;
  }
  return args;
}

async function loadInputs(analysisDir) {
  const missing = [];
  for (const name of REQUIRED_INPUTS) {
    try {
      await fs.access(path.join(analysisDir, name));
    } catch {
      missing.push(path.join('artifacts', CAMPAIGN, 'analysis', name));
    }
  }
  if (missing.length) {
    throw new Error(
      'Missing required EdgeTwinCal analysis inputs:\n - ' +
      missing.join('\n - ') +
      '\nGenerate gate_decision.json and failure_diagnosis.json before rerunning.',
    );
  }
  const loaded = {};
  for (const name of REQUIRED_INPUTS) {
    const absolute = path.join(analysisDir, name);
    loaded[name] = JSON.parse(await fs.readFile(absolute, 'utf8'));
  }
  return loaded;
}

function orderedVariants(variantMetrics) {
  const keys = Object.keys(variantMetrics || {});
  return keys.sort((left, right) => {
    const leftIndex = VARIANT_ORDER.indexOf(left);
    const rightIndex = VARIANT_ORDER.indexOf(right);
    if (leftIndex === -1 && rightIndex === -1) return left.localeCompare(right);
    if (leftIndex === -1) return 1;
    if (rightIndex === -1) return -1;
    return leftIndex - rightIndex;
  });
}

function asFinite(value) {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function safeRelativeGain(reference, candidate) {
  if (!Number.isFinite(reference) || !Number.isFinite(candidate) || reference === 0) {
    return null;
  }
  return (reference - candidate) / reference;
}

function buildDatasetVariantRows(aggregate) {
  const rows = [];
  for (const analysis of aggregate.analyses || []) {
    const apn = analysis.variant_metrics && analysis.variant_metrics.APN;
    const apnMse = apn ? asFinite(apn.mse) : null;
    for (const variant of orderedVariants(analysis.variant_metrics)) {
      const metrics = analysis.variant_metrics[variant] || {};
      const seed = (analysis.seed_descriptive || {})[variant] || {};
      rows.push({
        protocol: analysis.protocol,
        dataset: analysis.dataset,
        strict: Boolean(analysis.strict),
        inference: analysis.inference || '',
        reliable_group_ids: Boolean(analysis.reliable_group_ids),
        g2_status: (analysis.G2 || {}).status || '',
        g3_classification: (analysis.G3 || {}).classification || '',
        variant,
        mse: asFinite(metrics.mse),
        mae: asFinite(metrics.mae),
        n: asFinite(metrics.n),
        mse_mean: asFinite(seed.mse_mean),
        mse_std: asFinite(seed.mse_std),
        mae_mean: asFinite(seed.mae_mean),
        mae_std: asFinite(seed.mae_std),
        seed_count: asFinite(seed.seed_count),
        apn_mse: apnMse,
        relative_mse_gain_vs_apn: safeRelativeGain(apnMse, metrics.mse),
      });
    }
  }
  return rows;
}

function buildSeedRows(aggregate) {
  const rows = [];
  for (const analysis of aggregate.analyses || []) {
    const descriptive = analysis.seed_descriptive || {};
    const apnSeeds = (descriptive.APN || {}).seeds || {};
    for (const variant of orderedVariants(descriptive)) {
      const seeds = (descriptive[variant] || {}).seeds || {};
      const seedIds = Object.keys(seeds).sort((left, right) => Number(left) - Number(right));
      for (const seed of seedIds) {
        const metrics = seeds[seed] || {};
        const apnMse = asFinite((apnSeeds[seed] || {}).mse);
        rows.push({
          protocol: analysis.protocol,
          dataset: analysis.dataset,
          strict: Boolean(analysis.strict),
          variant,
          seed: Number.isFinite(Number(seed)) ? Number(seed) : seed,
          mse: asFinite(metrics.mse),
          mae: asFinite(metrics.mae),
          n: asFinite(metrics.n),
          apn_seed_mse: apnMse,
          relative_mse_gain_vs_apn: safeRelativeGain(apnMse, metrics.mse),
        });
      }
    }
  }
  return rows;
}

function buildPairedRows(aggregate) {
  const rows = [];
  for (const analysis of aggregate.analyses || []) {
    if (!analysis.strict) continue;
    for (const [reference, comparison] of Object.entries(analysis.comparisons || {})) {
      for (const metric of ['mse', 'mae']) {
        const values = ((comparison || {}).metrics || {})[metric] || {};
        rows.push({
          protocol: analysis.protocol,
          dataset: analysis.dataset,
          reference,
          metric,
          full_point: asFinite(values.full_point),
          reference_point: asFinite(values.reference_point),
          effect_full_minus_reference: asFinite(values.effect_full_minus_reference),
          effect_ci_low: asFinite(values.effect_ci_low),
          effect_ci_high: asFinite(values.effect_ci_high),
          relative_gain: asFinite(values.relative_gain),
          relative_gain_ci_low: asFinite(values.relative_gain_ci_low),
          relative_gain_ci_high: asFinite(values.relative_gain_ci_high),
          relative_loss_full_vs_reference: asFinite(
            values.relative_loss_full_vs_reference,
          ),
          relative_loss_ci_low: asFinite(values.relative_loss_ci_low),
          relative_loss_ci_high: asFinite(values.relative_loss_ci_high),
          p_full_not_better: asFinite(values.p_full_not_better),
          holm_adjusted_p: asFinite(values.holm_adjusted_p),
          holm_family: values.holm_family || '',
          resamples: asFinite(values.resamples),
        });
      }
    }
  }
  return rows;
}

function scalarType(value) {
  if (value === null) return 'null';
  if (Array.isArray(value)) return 'array';
  return typeof value;
}

function flattenLeaves(value, prefix = '', output = []) {
  if (value === null || typeof value !== 'object') {
    output.push({ path: prefix || '$', value, data_type: scalarType(value) });
    return output;
  }
  if (Array.isArray(value)) {
    if (value.length === 0) {
      output.push({ path: prefix, value: '[]', data_type: 'array' });
    } else {
      value.forEach((entry, index) => {
        flattenLeaves(entry, prefix + '[' + index + ']', output);
      });
    }
    return output;
  }
  const entries = Object.entries(value);
  if (entries.length === 0) {
    output.push({ path: prefix, value: '{}', data_type: 'object' });
  } else {
    for (const [key, entry] of entries) {
      const next = prefix ? prefix + '.' + key : key;
      flattenLeaves(entry, next, output);
    }
  }
  return output;
}

function buildGateRows(aggregate, gateDecision) {
  const rows = [];
  for (const leaf of flattenLeaves(gateDecision)) {
    rows.push({ scope: 'gate_decision', ...leaf });
  }
  for (const leaf of flattenLeaves(aggregate.gates || {})) {
    rows.push({ scope: 'confirmatory_aggregate.gates', ...leaf });
  }
  return rows.sort(
    (left, right) =>
      left.scope.localeCompare(right.scope) || left.path.localeCompare(right.path),
  );
}

function csvCell(value) {
  if (value === null || value === undefined) return '';
  const text = typeof value === 'boolean' ? (value ? 'true' : 'false') : String(value);
  if (/[",\r\n]/.test(text)) {
    return '"' + text.replaceAll('"', '""') + '"';
  }
  return text;
}

function toCsv(headers, rows) {
  const lines = [headers.map(csvCell).join(',')];
  for (const row of rows) {
    lines.push(headers.map((header) => csvCell(row[header])).join(','));
  }
  return lines.join('\r\n') + '\r\n';
}

async function sha256File(filePath) {
  const data = await fs.readFile(filePath);
  return crypto.createHash('sha256').update(data).digest('hex');
}

function cellValue(value) {
  if (value === null || value === undefined) return '';
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  return String(value);
}

function excelColumn(index) {
  let value = index + 1;
  let output = '';
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}

function writeMatrix(sheet, startRow, startCol, matrix) {
  if (!matrix.length || !matrix[0].length) return null;
  const range = sheet.getRangeByIndexes(
    startRow,
    startCol,
    matrix.length,
    matrix[0].length,
  );
  range.values = matrix;
  return range;
}

function styleTitle(sheet, width) {
  const range = sheet.getRangeByIndexes(0, 0, 1, width);
  range.format = {
    fill: COLORS.navy,
    font: { bold: true, color: COLORS.white, size: 16 },
    rowHeight: 30,
    verticalAlignment: 'center',
  };
}

function styleHeader(range) {
  range.format = {
    fill: COLORS.teal,
    font: { bold: true, color: COLORS.white },
    horizontalAlignment: 'center',
    verticalAlignment: 'center',
    wrapText: true,
    rowHeight: 34,
    borders: { preset: 'outside', style: 'thin', color: COLORS.navy },
  };
}

function styleData(range) {
  range.format = {
    font: { color: COLORS.text, size: 10 },
    verticalAlignment: 'center',
    borders: { preset: 'inside', style: 'thin', color: '#E2E8F0' },
  };
}

function applyAlternatingFill(sheet, startRow, endRow, startCol, colCount) {
  for (let row = startRow; row <= endRow; row += 2) {
    sheet.getRangeByIndexes(row, startCol, 1, colCount).format.fill = COLORS.paleGray;
  }
}

function setColumnWidths(sheet, widths) {
  widths.forEach((width, index) => {
    const column = excelColumn(index);
    sheet.getRange(column + ':' + column).format.columnWidth = width;
  });
}

function deriveVerdict(gateDecision) {
  return (
    gateDecision.verdict ||
    gateDecision.overall_verdict ||
    ((gateDecision.decision || {}).verdict) ||
    'UNKNOWN'
  );
}

function buildResultsSheet(workbook, aggregate, gateDecision, rows) {
  const sheet = workbook.worksheets.add('Results');
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [['EdgeTwinCal confirmatory lab results']];
  styleTitle(sheet, 26);
  const verdict = deriveVerdict(gateDecision);
  const strictCount = (aggregate.analyses || []).filter((item) => item.strict).length;
  sheet.getRange('A2:H3').values = [
    [
      'Formal verdict', verdict,
      'Complete manifests', (aggregate.input_audit || {}).complete_manifest_count || '',
      'Strict datasets', strictCount,
      'Bootstrap resamples', (aggregate.bootstrap || {}).resamples || '',
    ],
    [
      'Gate G2', ((aggregate.gates || {}).G2 || {}).status || '',
      'Gate G3', ((aggregate.gates || {}).G3 || {}).status || '',
      'Gate G4', ((aggregate.gates || {}).G4 || {}).status || '',
      'CI convention', 'raw percentile 95% CI',
    ],
  ];
  sheet.getRange('A2:H3').format = {
    fill: COLORS.paleBlue,
    font: { color: COLORS.text, size: 10 },
    wrapText: true,
    borders: { preset: 'outside', style: 'thin', color: COLORS.line },
  };
  for (const rangeName of ['A2:A3', 'C2:C3', 'E2:E3', 'G2:G3']) {
    sheet.getRange(rangeName).format.font = { bold: true, color: COLORS.navy };
  }

  const headers = [
    'Protocol', 'Dataset', 'Strict', 'Inference', 'Group IDs reliable',
    'G2 status', 'G3 classification', 'Variant', 'MSE', 'MAE', 'N',
    'Seed MSE mean', 'Seed MSE std', 'APN MSE', 'Relative MSE gain vs APN',
    'Seed count',
  ];
  const startRow = 4;
  writeMatrix(sheet, startRow, 0, [headers]);
  styleHeader(sheet.getRangeByIndexes(startRow, 0, 1, headers.length));
  const values = rows.map((row) => [
    row.protocol, row.dataset, row.strict, row.inference, row.reliable_group_ids,
    row.g2_status, row.g3_classification, row.variant, row.mse, row.mae, row.n,
    row.mse_mean, row.mse_std, '', '', row.seed_count,
  ]);
  writeMatrix(sheet, startRow + 1, 0, values);

  const analysisKeyToApnRow = new Map();
  rows.forEach((row, index) => {
    if (row.variant === 'APN') {
      analysisKeyToApnRow.set(
        row.protocol + '|' + row.dataset,
        startRow + 2 + index,
      );
    }
  });
  const formulas = rows.map((row, index) => {
    const excelRow = startRow + 2 + index;
    const apnRow = analysisKeyToApnRow.get(row.protocol + '|' + row.dataset);
    if (!apnRow) {
      throw new Error('Missing APN row for ' + row.protocol + '/' + row.dataset);
    }
    return [
      '=$I$' + apnRow,
      '=($N' + excelRow + '-$I' + excelRow + ')/$N' + excelRow,
    ];
  });
  sheet.getRangeByIndexes(startRow + 1, 13, rows.length, 2).formulas = formulas;
  const dataRange = sheet.getRangeByIndexes(startRow + 1, 0, rows.length, headers.length);
  styleData(dataRange);
  applyAlternatingFill(sheet, startRow + 1, startRow + rows.length, 0, headers.length);
  sheet.getRangeByIndexes(startRow + 1, 8, rows.length, 2).format.numberFormat = '0.000000';
  sheet.getRangeByIndexes(startRow + 1, 10, rows.length, 1).format.numberFormat = '#,##0';
  sheet.getRangeByIndexes(startRow + 1, 11, rows.length, 3).format.numberFormat = '0.000000';
  sheet.getRangeByIndexes(startRow + 1, 14, rows.length, 1).format.numberFormat = '0.00%';
  sheet.getRangeByIndexes(startRow + 1, 15, rows.length, 1).format.numberFormat = '#,##0';
  sheet.freezePanes.freezeRows(startRow + 1);

  const strictRows = {};
  rows.forEach((row, index) => {
    if (row.strict && (row.variant === 'APN' || row.variant === 'Full')) {
      strictRows[row.dataset + '|' + row.variant] = startRow + 2 + index;
    }
  });
  const strictDatasets = [...new Set(
    rows.filter((row) => row.strict).map((row) => row.dataset),
  )].sort();
  if (strictDatasets.length !== 2) {
    throw new Error(
      'Expected exactly two strict datasets for the main chart; found ' +
      strictDatasets.length,
    );
  }
  sheet.getRange('R4:T4').values = [['Dataset', 'APN MSE', 'Full MSE']];
  styleHeader(sheet.getRange('R4:T4'));
  const helperValues = strictDatasets.map((dataset) => [dataset, '', '']);
  sheet.getRange('R5:T6').values = helperValues;
  const helperFormulas = strictDatasets.map((dataset) => {
    const apnRow = strictRows[dataset + '|APN'];
    const fullRow = strictRows[dataset + '|Full'];
    if (!apnRow || !fullRow) {
      throw new Error('Missing strict APN/Full chart row for ' + dataset);
    }
    return ['=$I$' + apnRow, '=$I$' + fullRow];
  });
  sheet.getRange('S5:T6').formulas = helperFormulas;
  sheet.getRange('S5:T6').format.numberFormat = '0.000000';
  const chart = sheet.charts.add('bar', sheet.getRange('R4:T6'));
  chart.title = 'Strict test MSE: APN vs Full (lower is better)';
  chart.titleTextStyle.fontSize = 12;
  chart.hasLegend = true;
  chart.xAxis = { axisType: 'textAxis', textStyle: { fontSize: 10 } };
  chart.yAxis = { numberFormatCode: '0.000', textStyle: { fontSize: 9 } };
  chart.setPosition('R8', 'Z25');

  setColumnWidths(sheet, [
    18, 17, 9, 22, 15, 13, 18, 12,
    13, 13, 12, 15, 15, 13, 18, 12,
  ]);
  sheet.getRange('R:T').format.columnWidth = 15;
}

function buildSeedSheet(workbook, rows) {
  const sheet = workbook.worksheets.add('Seed metrics');
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [['Per-checkpoint descriptive metrics']];
  styleTitle(sheet, 10);
  sheet.getRange('A2').values = [[
    'Release-parity rows are seed-descriptive only; strict inference is reported in Paired CIs.',
  ]];
  sheet.getRange('A2:J2').format = {
    fill: COLORS.paleAmber,
    font: { italic: true, color: COLORS.text },
    wrapText: true,
  };
  const headers = [
    'Protocol', 'Dataset', 'Strict', 'Variant', 'Seed',
    'MSE', 'MAE', 'N', 'APN seed MSE', 'Relative MSE gain vs APN',
  ];
  const startRow = 3;
  writeMatrix(sheet, startRow, 0, [headers]);
  styleHeader(sheet.getRangeByIndexes(startRow, 0, 1, headers.length));
  const values = rows.map((row) => [
    row.protocol, row.dataset, row.strict, row.variant, row.seed,
    row.mse, row.mae, row.n, '', '',
  ]);
  writeMatrix(sheet, startRow + 1, 0, values);
  const apnRows = new Map();
  rows.forEach((row, index) => {
    if (row.variant === 'APN') {
      apnRows.set(
        row.protocol + '|' + row.dataset + '|' + row.seed,
        startRow + 2 + index,
      );
    }
  });
  const formulas = rows.map((row, index) => {
    const excelRow = startRow + 2 + index;
    const apnRow = apnRows.get(row.protocol + '|' + row.dataset + '|' + row.seed);
    if (!apnRow) {
      throw new Error(
        'Missing seed APN row for ' +
        row.protocol + '/' + row.dataset + '/' + row.seed,
      );
    }
    return [
      '=$F$' + apnRow,
      '=($I' + excelRow + '-$F' + excelRow + ')/$I' + excelRow,
    ];
  });
  sheet.getRangeByIndexes(startRow + 1, 8, rows.length, 2).formulas = formulas;
  const dataRange = sheet.getRangeByIndexes(startRow + 1, 0, rows.length, headers.length);
  styleData(dataRange);
  applyAlternatingFill(sheet, startRow + 1, startRow + rows.length, 0, headers.length);
  sheet.getRangeByIndexes(startRow + 1, 4, rows.length, 1).format.numberFormat = '0';
  sheet.getRangeByIndexes(startRow + 1, 5, rows.length, 2).format.numberFormat = '0.000000';
  sheet.getRangeByIndexes(startRow + 1, 7, rows.length, 1).format.numberFormat = '#,##0';
  sheet.getRangeByIndexes(startRow + 1, 8, rows.length, 1).format.numberFormat = '0.000000';
  sheet.getRangeByIndexes(startRow + 1, 9, rows.length, 1).format.numberFormat = '0.00%';
  sheet.freezePanes.freezeRows(startRow + 1);
  setColumnWidths(sheet, [18, 17, 9, 12, 10, 14, 14, 12, 15, 20]);
}

function buildPairedSheet(workbook, rows) {
  const sheet = workbook.worksheets.add('Paired CIs');
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [['Strict paired bootstrap comparisons']];
  styleTitle(sheet, 19);
  sheet.getRange('A2').values = [[
    'Intervals are raw crossed group-by-checkpoint percentile 95% CIs. ' +
    'Holm correction applies to the one-sided bootstrap p-value only.',
  ]];
  sheet.getRange('A2:S2').format = {
    fill: COLORS.paleAmber,
    font: { italic: true, color: COLORS.text },
    wrapText: true,
  };
  const headers = [
    'Protocol', 'Dataset', 'Reference', 'Metric', 'Full point', 'Reference point',
    'Effect Full - reference', 'Effect CI low', 'Effect CI high', 'Relative gain',
    'Gain CI low', 'Gain CI high', 'Relative loss', 'Loss CI low', 'Loss CI high',
    'p(Full not better)', 'Holm adjusted p', 'Holm family', 'Resamples',
  ];
  const startRow = 3;
  writeMatrix(sheet, startRow, 0, [headers]);
  styleHeader(sheet.getRangeByIndexes(startRow, 0, 1, headers.length));
  const values = rows.map((row) => [
    row.protocol, row.dataset, row.reference, row.metric,
    row.full_point, row.reference_point, row.effect_full_minus_reference,
    row.effect_ci_low, row.effect_ci_high, row.relative_gain,
    row.relative_gain_ci_low, row.relative_gain_ci_high,
    row.relative_loss_full_vs_reference, row.relative_loss_ci_low,
    row.relative_loss_ci_high, row.p_full_not_better, row.holm_adjusted_p,
    row.holm_family, row.resamples,
  ]);
  writeMatrix(sheet, startRow + 1, 0, values);
  const dataRange = sheet.getRangeByIndexes(startRow + 1, 0, rows.length, headers.length);
  styleData(dataRange);
  applyAlternatingFill(sheet, startRow + 1, startRow + rows.length, 0, headers.length);
  sheet.getRangeByIndexes(startRow + 1, 4, rows.length, 5).format.numberFormat = '0.000000';
  sheet.getRangeByIndexes(startRow + 1, 9, rows.length, 6).format.numberFormat = '0.00%';
  sheet.getRangeByIndexes(startRow + 1, 15, rows.length, 2).format.numberFormat = '0.0000';
  sheet.getRangeByIndexes(startRow + 1, 18, rows.length, 1).format.numberFormat = '#,##0';
  sheet.freezePanes.freezeRows(startRow + 1);
  setColumnWidths(sheet, [
    18, 17, 12, 10, 14, 15, 18, 14, 14, 14,
    14, 14, 14, 14, 14, 17, 16, 13, 12,
  ]);
}

function buildGateSheet(workbook, gateDecision, rows) {
  const sheet = workbook.worksheets.add('Gate audit');
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [['Formal gate decision audit']];
  styleTitle(sheet, 4);
  const verdict = deriveVerdict(gateDecision);
  sheet.getRange('A2:B2').values = [['Overall verdict', verdict]];
  sheet.getRange('A2:B2').format = {
    fill: verdict === 'ABANDON' ? COLORS.paleRed : COLORS.paleGreen,
    font: { bold: true, color: COLORS.text },
    borders: { preset: 'outside', style: 'thin', color: COLORS.line },
  };
  const headers = ['Scope', 'Path', 'Value', 'Data type'];
  const startRow = 3;
  writeMatrix(sheet, startRow, 0, [headers]);
  styleHeader(sheet.getRangeByIndexes(startRow, 0, 1, headers.length));
  const values = rows.map((row) => [
    row.scope, row.path, cellValue(row.value), row.data_type,
  ]);
  writeMatrix(sheet, startRow + 1, 0, values);
  const dataRange = sheet.getRangeByIndexes(startRow + 1, 0, rows.length, headers.length);
  styleData(dataRange);
  dataRange.format.wrapText = true;
  applyAlternatingFill(sheet, startRow + 1, startRow + rows.length, 0, headers.length);
  const statusRange = sheet.getRangeByIndexes(startRow + 1, 2, rows.length, 1);
  statusRange.conditionalFormats.add('containsText', {
    text: 'PASS',
    format: { fill: COLORS.paleGreen, font: { color: '#166534', bold: true } },
  });
  statusRange.conditionalFormats.add('containsText', {
    text: 'FAIL',
    format: { fill: COLORS.paleRed, font: { color: '#991B1B', bold: true } },
  });
  statusRange.conditionalFormats.add('containsText', {
    text: 'BLOCKED',
    format: { fill: COLORS.paleAmber, font: { color: '#92400E', bold: true } },
  });
  statusRange.conditionalFormats.add('containsText', {
    text: 'ABANDON',
    format: { fill: COLORS.paleRed, font: { color: '#991B1B', bold: true } },
  });
  sheet.freezePanes.freezeRows(startRow + 1);
  setColumnWidths(sheet, [28, 54, 58, 14]);
}

async function buildProvenanceSheet(
  workbook,
  analysisDir,
  aggregate,
  gateDecision,
  failureDiagnosis,
) {
  const sheet = workbook.worksheets.add('Provenance');
  sheet.showGridLines = false;
  sheet.getRange('A1').values = [['Inputs, hashes, and analysis provenance']];
  styleTitle(sheet, 5);
  const sourceHeaders = ['Source', 'Relative path', 'SHA256', 'Bytes', 'Schema / verdict'];
  writeMatrix(sheet, 2, 0, [sourceHeaders]);
  styleHeader(sheet.getRange('A3:E3'));
  const sourceRows = [];
  for (const name of REQUIRED_INPUTS) {
    const absolute = path.join(analysisDir, name);
    const stat = await fs.stat(absolute);
    let label = '';
    if (name === 'confirmatory_aggregate.json') {
      label = aggregate.schema_version || '';
    } else if (name === 'gate_decision.json') {
      label = deriveVerdict(gateDecision);
    } else {
      label = failureDiagnosis.schema_version || failureDiagnosis.verdict || '';
    }
    sourceRows.push([
      name.replace('.json', ''),
      path.join('artifacts', CAMPAIGN, 'analysis', name).replaceAll('\\', '/'),
      await sha256File(absolute),
      stat.size,
      label,
    ]);
  }
  writeMatrix(sheet, 3, 0, sourceRows);
  styleData(sheet.getRange('A4:E6'));
  sheet.getRange('D4:D6').format.numberFormat = '#,##0';

  const auditHeaders = ['Source', 'Path', 'Value', 'Data type'];
  writeMatrix(sheet, 7, 0, [auditHeaders]);
  styleHeader(sheet.getRange('A8:D8'));
  const auditRows = [];
  for (const leaf of flattenLeaves({
    schema_version: aggregate.schema_version,
    bootstrap: aggregate.bootstrap,
    input_audit: aggregate.input_audit,
    holm_families: aggregate.holm_families,
  })) {
    auditRows.push(['confirmatory_aggregate', leaf.path, cellValue(leaf.value), leaf.data_type]);
  }
  const failureLeaves = flattenLeaves(failureDiagnosis);
  const maxFailureLeaves = 500;
  for (const leaf of failureLeaves.slice(0, maxFailureLeaves)) {
    auditRows.push(['failure_diagnosis', leaf.path, cellValue(leaf.value), leaf.data_type]);
  }
  if (failureLeaves.length > maxFailureLeaves) {
    auditRows.push([
      'failure_diagnosis',
      '_workbook_note',
      String(failureLeaves.length - maxFailureLeaves) +
      ' additional scalar leaves remain in the hashed source JSON.',
      'string',
    ]);
  }
  writeMatrix(sheet, 8, 0, auditRows);
  const auditRange = sheet.getRangeByIndexes(8, 0, auditRows.length, 4);
  styleData(auditRange);
  auditRange.format.wrapText = true;
  applyAlternatingFill(sheet, 8, 7 + auditRows.length, 0, 4);
  sheet.freezePanes.freezeRows(8);
  setColumnWidths(sheet, [28, 58, 72, 14, 24]);
}

async function writeCsvOutputs(outputDir, datasetRows, seedRows, pairedRows, gateRows) {
  const datasetHeaders = [
    'protocol', 'dataset', 'strict', 'inference', 'reliable_group_ids',
    'g2_status', 'g3_classification', 'variant', 'mse', 'mae', 'n',
    'mse_mean', 'mse_std', 'mae_mean', 'mae_std', 'seed_count',
    'apn_mse', 'relative_mse_gain_vs_apn',
  ];
  const seedHeaders = [
    'protocol', 'dataset', 'strict', 'variant', 'seed',
    'mse', 'mae', 'n', 'apn_seed_mse', 'relative_mse_gain_vs_apn',
  ];
  const pairedHeaders = [
    'protocol', 'dataset', 'reference', 'metric', 'full_point', 'reference_point',
    'effect_full_minus_reference', 'effect_ci_low', 'effect_ci_high',
    'relative_gain', 'relative_gain_ci_low', 'relative_gain_ci_high',
    'relative_loss_full_vs_reference', 'relative_loss_ci_low',
    'relative_loss_ci_high', 'p_full_not_better', 'holm_adjusted_p',
    'holm_family', 'resamples',
  ];
  const gateHeaders = ['scope', 'path', 'value', 'data_type'];
  const outputs = [
    ['dataset_variant_summary.csv', datasetHeaders, datasetRows],
    ['seed_summary.csv', seedHeaders, seedRows],
    ['paired_comparisons.csv', pairedHeaders, pairedRows],
    ['gate_summary.csv', gateHeaders, gateRows],
  ];
  for (const [name, headers, rows] of outputs) {
    await fs.writeFile(
      path.join(outputDir, name),
      toCsv(headers, rows),
      'utf8',
    );
  }
}

async function verifyAndRender(workbook, qaDir) {
  const checks = [
    ['Results', 'A1:Z28'],
    ['Seed metrics', 'A1:J36'],
    ['Paired CIs', 'A1:S48'],
    ['Gate audit', 'A1:D45'],
    ['Provenance', 'A1:E45'],
  ];
  for (const [sheetName, range] of checks) {
    const inspected = await workbook.inspect({
      kind: 'table',
      range: sheetName + '!' + range,
      include: 'values,formulas',
      tableMaxRows: 12,
      tableMaxCols: 20,
      maxChars: 5000,
    });
    console.log(JSON.stringify({
      qa: 'inspect',
      sheet: sheetName,
      preview: inspected.ndjson,
    }));
    if (qaDir) {
      const rendered = await workbook.render({
        sheetName,
        range,
        scale: 1,
        format: 'png',
      });
      const safeName = sheetName.toLowerCase().replaceAll(' ', '_') + '.png';
      await fs.writeFile(
        path.join(qaDir, safeName),
        new Uint8Array(await rendered.arrayBuffer()),
      );
    }
  }
  const errors = await workbook.inspect({
    kind: 'match',
    searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A',
    options: { useRegex: true, maxResults: 300 },
    summary: 'final formula error scan',
    maxChars: 5000,
  });
  console.log(JSON.stringify({ qa: 'formula_errors', scan: errors.ndjson }));
  const errorText = errors.ndjson || '';
  if (/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A/.test(errorText)) {
    throw new Error('Formula error scan returned one or more spreadsheet errors.');
  }
}

async function main() {
  const args = parseArgs(process.argv);
  const inputs = await loadInputs(args.analysisDir);
  const aggregate = inputs['confirmatory_aggregate.json'];
  const gateDecision = inputs['gate_decision.json'];
  const failureDiagnosis = inputs['failure_diagnosis.json'];
  if (!Array.isArray(aggregate.analyses) || aggregate.analyses.length === 0) {
    throw new Error('confirmatory_aggregate.json has no analyses.');
  }
  if ((aggregate.input_audit || {}).complete_manifest_count !== 180) {
    throw new Error(
      'Expected 180 complete manifests; found ' +
      String((aggregate.input_audit || {}).complete_manifest_count),
    );
  }

  await fs.mkdir(args.outputDir, { recursive: true });
  if (args.qaDir) await fs.mkdir(args.qaDir, { recursive: true });
  const datasetRows = buildDatasetVariantRows(aggregate);
  const seedRows = buildSeedRows(aggregate);
  const pairedRows = buildPairedRows(aggregate);
  const gateRows = buildGateRows(aggregate, gateDecision);
  await writeCsvOutputs(
    args.outputDir,
    datasetRows,
    seedRows,
    pairedRows,
    gateRows,
  );

  let artifactTool;
  try {
    artifactTool = await import('@oai/artifact-tool');
  } catch (error) {
    throw new Error(
      'Bundled @oai/artifact-tool is unavailable. ' +
      'Run from the Codex loader-provided Node runtime and node_modules junction. ' +
      String(error),
    );
  }
  const { SpreadsheetFile, Workbook } = artifactTool;
  const workbook = Workbook.create();
  buildResultsSheet(workbook, aggregate, gateDecision, datasetRows);
  buildSeedSheet(workbook, seedRows);
  buildPairedSheet(workbook, pairedRows);
  buildGateSheet(workbook, gateDecision, gateRows);
  await buildProvenanceSheet(
    workbook,
    args.analysisDir,
    aggregate,
    gateDecision,
    failureDiagnosis,
  );
  await verifyAndRender(workbook, args.qaDir);

  const outputPath = path.join(args.outputDir, 'EdgeTwinCal_lab_results.xlsx');
  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(outputPath);
  console.log(JSON.stringify({
    status: 'ok',
    output: outputPath,
    sheets: SHEET_NAMES,
    row_counts: {
      dataset_variant_summary: datasetRows.length,
      seed_summary: seedRows.length,
      paired_comparisons: pairedRows.length,
      gate_summary: gateRows.length,
    },
  }));
}

main().catch((error) => {
  console.error('EdgeTwinCal table build failed: ' + error.message);
  process.exitCode = 1;
});
