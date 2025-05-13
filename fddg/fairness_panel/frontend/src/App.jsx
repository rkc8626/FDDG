import React, { useEffect, useState, useCallback, useRef } from "react";
import {
  createTheme,
  ThemeProvider,
  CssBaseline,
  AppBar,
  Toolbar,
  Container,
  Box,
  IconButton,
  Typography,
  Paper,
  Button,
  TextField,
  CircularProgress,
  Divider,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Modal,
  Tabs,
  Tab,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
} from "@mui/material";
import LightModeIcon from "@mui/icons-material/LightMode";
import DarkModeIcon from "@mui/icons-material/DarkMode";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend as ChartLegend,
  TimeScale,
  RadialLinearScale,
  Filler,
} from "chart.js";
import zoomPlugin from "chartjs-plugin-zoom";
import { Line, Radar, Scatter } from "react-chartjs-2";
import { LazyLog } from "react-lazylog";
import { io } from 'socket.io-client';

// ============== REGISTER CHART.JS + PLUGIN ==============
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  ChartLegend,
  TimeScale,
  zoomPlugin,
  RadialLinearScale,
  Filler
);

/**
 * Custom hook: fetch /scalars + /training_state, handle config logic,
 * store chart zoom in localStorage
 */
function useDashboardData() {
  const [scalars, setScalars] = useState({});
  const [trainingState, setTrainingState] = useState(null);
  const [currentConfig, setCurrentConfig] = useState(null);
  const [inputConfig, setInputConfig] = useState({
    batch_size: "",
    learning_rate: "",
    running: false
  });
  const [isChanging, setIsChanging] = useState(false);
  const [error, setError] = useState(null);
  const [socket, setSocket] = useState(null);
  const [scalarData, setScalarData] = useState({});
  const [inputInitialized, setInputInitialized] = useState(false);
  const [connectionState, setConnectionState] = useState({
    isConnected: false,
    error: null,
    reconnectAttempt: 0
  });

  // Zoom/pan ranges stored in localStorage
  const [zoomRanges, setZoomRanges] = useState({});

  // Initialize input values ONLY ONCE when current config is first loaded
  useEffect(() => {
    if (currentConfig && !inputInitialized) {
      setInputConfig({
        batch_size: String(currentConfig.batch_size),
        learning_rate: String(currentConfig.learning_rate),
        running: currentConfig.running
      });
      setInputInitialized(true);
    }
  }, [currentConfig, inputInitialized]);

  // Initialize WebSocket connection and fetch initial data
  useEffect(() => {
    let reconnectTimer;

    const connectSocket = () => {
      const newSocket = io('http://localhost:5100', {
        transports: ['websocket'],
        reconnection: false,  // We'll handle reconnection manually
      });

      newSocket.on('connect', () => {
        console.log('Connected to server');
        setConnectionState(prev => ({
          ...prev,
          isConnected: true,
          error: null,
          reconnectAttempt: 0
        }));
        newSocket.emit('request_state');
      });

      newSocket.on('connect_error', (error) => {
        console.log('Connection Error:', error);
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          error: 'Failed to connect to server',
          reconnectAttempt: prev.reconnectAttempt + 1
        }));
      });

      newSocket.on('disconnect', (reason) => {
        console.log('Disconnected:', reason);
        setConnectionState(prev => ({
          ...prev,
          isConnected: false,
          error: `Disconnected: ${reason}`,
          reconnectAttempt: prev.reconnectAttempt + 1
        }));
      });

      newSocket.on('state_update', (data) => {
        console.log('Received state update:', data);
        setTrainingState(data.state);
        setCurrentConfig(data.config);
      });

      newSocket.on('config_updated', (response) => {
        console.log('Config update response:', response);
        if (response.status === 'error') {
          setError(response.message);
        }
        setIsChanging(false);
      });

      newSocket.on('state_error', (error) => {
        setError(error.message);
      });

      newSocket.on('scalar_update', (newData) => {
        setScalarData(newData);
      });

      setSocket(newSocket);
    };

    const startReconnectTimer = () => {
      reconnectTimer = setInterval(() => {
        if (!connectionState.isConnected) {
          console.log('Attempting to reconnect...');
          connectSocket();
        }
      }, 5000);
    };

    connectSocket();
    startReconnectTimer();

    return () => {
      clearInterval(reconnectTimer);
      if (socket) socket.disconnect();
    };
  }, [connectionState.isConnected]);

  // Fetch scalars separately
  useEffect(() => {
    fetchScalars();
  }, []);

  // Add auto-refresh for scalar data
  useEffect(() => {
    // Initial fetch
    fetchScalars();

    // Set up interval for auto-refresh
    const intervalId = setInterval(() => {
      console.log("Auto-refreshing scalar data...");
      fetchScalars();
    }, 10000); // Refresh every 10 seconds

    // Clean up on unmount
    return () => clearInterval(intervalId);
  }, []);

  // == Zoom from localStorage ==
  useEffect(() => {
    const saved = localStorage.getItem("chartZoomRanges");
    if (saved) {
      try {
        setZoomRanges(JSON.parse(saved));
      } catch (err) {
        console.error("Failed to parse chartZoomRanges:", err);
      }
    }
  }, []);
  useEffect(() => {
    localStorage.setItem("chartZoomRanges", JSON.stringify(zoomRanges));
  }, [zoomRanges]);

  function fetchScalars() {
    fetch("http://localhost:5100/scalars")
      .then((r) => r.json())
      .then((data) => setScalars(data))
      .catch((err) => setError(err.message));
  }

  const handleInputChange = (key, value) => {
    if (key === 'learning_rate' && !trainingState?.learning_rate) {
      console.log('Cannot modify learning rate before training starts');
      return;
    }

    setInputConfig(prev => ({
      ...prev,
      [key]: value
    }));
    setIsChanging(true);
  };

  const applyChanges = () => {
    if (!socket) return;

    const newConfig = {
      ...currentConfig,
      batch_size: parseInt(inputConfig.batch_size),
      lr: parseFloat(inputConfig.learning_rate),
      running: inputConfig.running
    };

    socket.emit('update_config', newConfig);
  };

  const handleStop = () => {
    if (!socket) return;
    setInputConfig(prev => ({ ...prev, running: false }));
    socket.emit('update_config', { ...currentConfig, running: false });
  };

  const handleStart = () => {
    if (!socket) return;
    setInputConfig(prev => ({ ...prev, running: true }));
    socket.emit('update_config', { ...currentConfig, running: true });
  };

  // == Zoom/Pan events ==
  function handleZoomPanComplete(metricKey, chart) {
    const xScale = chart.scales.x;
    const yScale = chart.scales.y;
    setZoomRanges((prev) => ({
      ...prev,
      [metricKey]: {
        xMin: xScale.min,
        xMax: xScale.max,
        yMin: yScale.min,
        yMax: yScale.max,
      },
    }));
  }
  function getChartOptions(metricKey) {
    const st = zoomRanges[metricKey];
    let base = {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { position: "top" },
        title: { display: false },
        zoom: {
          zoom: {
            wheel: { enabled: true }, // mouse wheel
            pinch: { enabled: true }, // pinch gesture
            mode: "xy",
          },
          pan: {
            enabled: true,
            mode: "xy",
          },
          onZoomComplete: ({ chart }) => handleZoomPanComplete(metricKey, chart),
          onPanComplete: ({ chart }) => handleZoomPanComplete(metricKey, chart),
        },
      },
      scales: {
        x: { type: "linear", title: { display: true, text: "Step" } },
        y: { title: { display: true, text: "Value" } },
      },
    };
    if (st) {
      base.scales.x.min = st.xMin;
      base.scales.x.max = st.xMax;
      base.scales.y.min = st.yMin;
      base.scales.y.max = st.yMax;
    }
    return base;
  }

  // Helper to check if changes are pending
  const isPending = useCallback(() => {
    if (!trainingState || !currentConfig) return false;

    // Compare state with config to see if simulator has caught up
    return trainingState.running !== currentConfig.running ||
           String(trainingState.batch_size) !== String(currentConfig.batch_size) ||
           (String(trainingState.hparams?.lr) !== String(currentConfig.learning_rate) &&
            currentConfig.learning_rate &&
            String(currentConfig.learning_rate) !== "" &&
            trainingState.running);
  }, [trainingState, currentConfig]);

  // Update chart data when scalarData changes
  useEffect(() => {
    if (Object.keys(scalarData).length > 0) {
      setScalars(scalarData);
    }
  }, [scalarData]);

  return {
    scalars,
    trainingState,
    currentConfig,
    inputConfig,
    isChanging,
    error,
    connectionState,
    isPending,
    handleInputChange,
    applyChanges,
    getChartOptions,
    handleStop,
    handleStart,
  };
}

// Add this helper function to group metrics by prefix
function groupMetricsByPrefix(scalars) {
  return scalars;
}

// Add new component for t-SNE visualization
function TrainingVisualization({ data, sensitiveColorMap }) {
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [processedData, setProcessedData] = useState(null);
  const [colorMode, setColorMode] = useState('sensitive'); // Only declare this ONCE

  useEffect(() => {
    setIsLoading(true);
    setError(null);

    try {
      if (!data) {
        setError("No data available");
        setIsLoading(false);
        return;
      }
      if (!data.points || !Array.isArray(data.points) || data.points.length === 0) {
        setError("No points data available");
        setIsLoading(false);
        return;
      }
      if (!data.sensitive || data.sensitive.length !== data.points.length) {
        setError("Invalid data format: mismatched array lengths");
        setIsLoading(false);
        return;
      }

      // Color and shape palettes
      const colorList = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0', '#9966FF', '#FF9F40', '#8BC34A', '#E91E63', '#00BCD4', '#CDDC39'
      ];
      const markerList = ['circle', 'triangle', 'rect', 'star', 'cross', 'rectRounded', 'rectRot', 'dash', 'line'];

      let groupValues, groupKey, groupSet, groupGuide, datasets;
      if (colorMode === 'sensitive') {
        groupValues = data.sensitive;
        groupSet = data.sensitive_set || Array.from(new Set(data.sensitive));
        groupKey = 'Sensitive';
        groupGuide = data.tsne_fairness_guide || {};
        datasets = groupSet.map((group, idx) => ({
          label: `${groupKey} ${group}`,
          data: data.points.filter((_, i) => groupValues[i] === group),
          backgroundColor: colorList[idx % colorList.length],
          pointStyle: 'circle',
          pointRadius: 5,
          pointHoverRadius: 7,
          parsing: false,
          normalized: true
        }));
      } else if (colorMode === 'label') {
        groupValues = data.labels;
        groupSet = data.labels_set || Array.from(new Set(data.labels));
        groupKey = 'Label';
        groupGuide = {};
        data.labels_set?.forEach(lab => { groupGuide[lab] = data.labels.filter(l => l === lab).length; });
        datasets = groupSet.map((group, idx) => ({
          label: `${groupKey} ${group}`,
          data: data.points.filter((_, i) => groupValues[i] === group),
          backgroundColor: colorList[idx % colorList.length],
          pointStyle: 'circle',
          pointRadius: 5,
          pointHoverRadius: 7,
          parsing: false,
          normalized: true
        }));
      } else if (colorMode === 'environment') {
        groupValues = data.environments;
        groupSet = Array.from(new Set(data.environments));
        groupKey = 'Env';
        groupGuide = {};
        groupSet.forEach(env => { groupGuide[env] = data.environments.filter(e => e === env).length; });
        datasets = groupSet.map((group, idx) => ({
          label: `${groupKey} ${group}`,
          data: data.points.filter((_, i) => groupValues[i] === group),
          backgroundColor: colorList[idx % colorList.length],
          pointStyle: 'circle',
          pointRadius: 5,
          pointHoverRadius: 7,
          parsing: false,
          normalized: true
        }));
      } else if (colorMode === 'split') {
        groupValues = data.split_types;
        groupSet = Array.from(new Set(data.split_types));
        groupKey = 'Split';
        groupGuide = {};
        groupSet.forEach(split => { groupGuide[split] = data.split_types.filter(s => s === split).length; });
        datasets = groupSet.map((group, idx) => ({
          label: `${groupKey} ${group}`,
          data: data.points.filter((_, i) => groupValues[i] === group),
          backgroundColor: colorList[idx % colorList.length],
          pointStyle: 'circle',
          pointRadius: 5,
          pointHoverRadius: 7,
          parsing: false,
          normalized: true
        }));
      } else if (colorMode === 'class_sensitive') {
        // Dual encoding: color by predicted class, shape by sensitive
        const predLabels = data.predicted_labels && data.predicted_labels.length === data.points.length
          ? data.predicted_labels
          : data.labels;
        const classSet = data.predicted_labels_set && data.predicted_labels_set.length > 0
          ? data.predicted_labels_set
          : (data.labels_set || Array.from(new Set(predLabels)));
        const sensSet = data.sensitive_set || Array.from(new Set(data.sensitive));
        groupKey = 'PredClass+Sensitive';
        groupGuide = {};
        datasets = [];
        classSet.forEach((cls, cidx) => {
          sensSet.forEach((sens, sidx) => {
            const points = data.points.filter((_, i) => predLabels[i] === cls && data.sensitive[i] === sens);
            if (points.length > 0) {
              const label = `Class ${cls}, Sensitive ${sens}`;
              groupGuide[label] = points.length;
              datasets.push({
                label,
                data: points,
                backgroundColor: colorList[cidx % colorList.length],
                pointStyle: markerList[sidx % markerList.length],
                pointRadius: 6,
                pointHoverRadius: 8,
                parsing: false,
                normalized: true
              });
            }
          });
        });
      }

      setProcessedData({ datasets, groupKey, groupGuide, colorMode });
      setIsLoading(false);
    } catch (err) {
      setError(err.message || "Error processing visualization data");
      setIsLoading(false);
    }
  }, [data, sensitiveColorMap, colorMode]);

  if (isLoading) {
    return (
      <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', alignItems: 'center', height: 400 }}>
        <CircularProgress />
      </Box>
    );
  }
  if (error) {
    return (
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', height: 400 }}>
        <Typography color="error" gutterBottom>
          {error}
        </Typography>
        <Typography variant="body2" color="text.secondary">
          The visualization will automatically update when data becomes available.
        </Typography>
      </Box>
    );
  }
  if (!processedData || !processedData.datasets || processedData.datasets.length === 0) {
    return (
      <Box sx={{ p: 2, display: 'flex', flexDirection: 'column', alignItems: 'center', height: 400 }}>
        <Typography variant="body1" gutterBottom>
          No visualization data available
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Please wait for the training to generate enough data points.
        </Typography>
      </Box>
    );
  }

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    elements: {
      point: { hoverRadius: 5, hoverBorderWidth: 0 }
    },
    scales: {
      x: {
        type: 'linear',
        position: 'bottom',
        title: { display: true, text: 't-SNE 1 (arbitrary units)' },
        ticks: { maxTicksLimit: 10 }
      },
      y: {
        title: { display: true, text: 't-SNE 2 (arbitrary units)' },
        ticks: { maxTicksLimit: 10 }
      }
    },
    plugins: {
      legend: { position: 'top' },
      title: { display: true, text: `t-SNE: Representation by ${processedData.groupKey}` },
      tooltip: {
        enabled: true,
        mode: 'nearest',
        intersect: false,
        position: 'nearest',
        callbacks: {
          label: function(context) {
            const point = context.raw;
            const dataset = context.dataset;
            const idx = context.dataIndex;
            let origIdx = -1;
            // For class+sens, try to find the original index
            if (processedData.colorMode === 'class_sensitive') {
              for (let i = 0, count = 0; i < data.points.length; ++i) {
                if (
                  `Class ${data.labels[i]}, Sensitive ${data.sensitive[i]}` === dataset.label &&
                  count === idx
                ) {
                  origIdx = i;
                  break;
                }
                if (`Class ${data.labels[i]}, Sensitive ${data.sensitive[i]}` === dataset.label) {
                  count++;
                }
              }
            } else {
              for (let i = 0, count = 0; i < data.points.length; ++i) {
                if (
                  String(dataset.label).endsWith(String(processedData.groupKey + ' ' + (data[processedData.groupKey.toLowerCase() + 's']?.[i] ?? '')))
                  && count === idx
                ) {
                  origIdx = i;
                  break;
                }
                if (
                  String(dataset.label).endsWith(String(processedData.groupKey + ' ' + (data[processedData.groupKey.toLowerCase() + 's']?.[i] ?? '')))
                ) {
                  count++;
                }
              }
            }
            return [
              processedData.colorMode === 'class_sensitive'
                ? `Predicted: ${data.predicted_labels?.[origIdx] ?? data.labels?.[origIdx]}, Sensitive: ${data.sensitive?.[origIdx]}`
                : `${processedData.groupKey}: ${dataset.label.replace(processedData.groupKey + ' ', '')}`,
              `True Label: ${data.labels?.[origIdx]}`,
              `Sensitive: ${data.sensitive?.[origIdx]}`,
              `Env: ${data.environments?.[origIdx]}`,
              `Position: (${(point?.x || 0).toFixed(2)}, ${(point?.y || 0).toFixed(2)})`
            ];
          }
        }
      }
    },
    devicePixelRatio: 1,
    interaction: { mode: 'nearest', axis: 'xy', intersect: false }
  };

  // Custom legend for groups
  const legend = (
    <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 2, mt: 1, mb: 2 }}>
      {processedData.datasets.map(ds => (
        <Box key={ds.label} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Box sx={{ width: 16, height: 16, backgroundColor: ds.backgroundColor, borderRadius: ds.pointStyle === 'circle' ? '50%' : '0', border: ds.pointStyle !== 'circle' ? '2px solid #333' : 'none', display: 'inline-block' }} />
          <Typography variant="body2">{ds.label} ({processedData.groupGuide?.[ds.label] || ds.data.length}) {processedData.colorMode === 'class_sensitive' ? ` [${ds.pointStyle}]` : ''}</Typography>
        </Box>
      ))}
    </Box>
  );

  // Dropdown for color mode
  const colorModeSelector = (
    <FormControl size="small" sx={{ minWidth: 180, mb: 1 }}>
      <InputLabel id="color-mode-label">Color By</InputLabel>
      <Select
        labelId="color-mode-label"
        id="color-mode-select"
        value={colorMode}
        label="Color By"
        onChange={e => setColorMode(e.target.value)}
      >
        <MenuItem value="sensitive">Sensitive Attribute</MenuItem>
        <MenuItem value="label">Class Label</MenuItem>
        <MenuItem value="environment">Environment</MenuItem>
        <MenuItem value="split">Train/Test Split</MenuItem>
        <MenuItem value="class_sensitive">Class + Sensitive</MenuItem>
      </Select>
    </FormControl>
  );

  return (
    <Box sx={{ height: 440, width: '100%', position: 'relative', mb: 12, pb: 3 }}>
      {colorModeSelector}
      {legend}
      <Scatter
        data={processedData}
        options={options}
        fallbackContent={
          <Typography>
            Unable to render visualization. Please try refreshing the page.
          </Typography>
        }
      />
      <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
        Note: t-SNE axes are arbitrary. Only the relative position and grouping of points is meaningful.
      </Typography>
    </Box>
  );
}

// Modify MetricsTabs to include the visualization
function MetricsTabs({ scalars, getChartOptions, trainingState }) {
  const [currentTab, setCurrentTab] = useState(0);
  const [visualizationData, setVisualizationData] = useState(null);
  const prevRunningRef = useRef(false);

  // Fetch visualization data periodically
  useEffect(() => {
    const fetchVisualization = async () => {
      try {
        const response = await fetch('http://localhost:5100/representations');
        const data = await response.json();
        if (!data.error) {
          setVisualizationData(data);
        }
      } catch (error) {
        console.error('Error fetching visualization data:', error);
      }
    };
    fetchVisualization();
    const interval = setInterval(fetchVisualization, 5000);
    return () => clearInterval(interval);
  }, []);

  // Clear t-SNE data when training starts
  useEffect(() => {
    if (trainingState?.running && !prevRunningRef.current) {
      setVisualizationData(null);
    }
    prevRunningRef.current = trainingState?.running;
  }, [trainingState?.running]);

  const groups = Object.keys(scalars).filter(group => group !== 'radar');

  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };

  const renderRadarChart = () => {
    const radarData = scalars.radar;
    console.log("Radar data:", radarData); // Debug log

    if (!radarData || !radarData.is_radar) {
      console.log("No radar data found or not marked as radar"); // Debug log
      return (
        <Box sx={{ mb: 4, mt: 2 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
            Multi-Metric Radar
          </Typography>
          <Typography>
            Waiting for radar metrics data... Please ensure your model is generating the required metrics.
          </Typography>
        </Box>
      );
    }

    // Get latest values for each metric
    const lastValues = {};
    Object.entries(radarData.values).forEach(([metricName, values]) => {
      console.log(`Processing ${metricName} with ${values.length} values`); // Debug log
      if (values.length > 0) {
        // Sort by step and get the latest value
        const sortedValues = [...values].sort((a, b) => a.step - b.step);
        lastValues[metricName] = sortedValues[sortedValues.length - 1].value;
        console.log(`Latest value for ${metricName}: ${lastValues[metricName]}`); // Debug log
      } else {
        // Default value if no data
        lastValues[metricName] = 0;
        console.log(`No data for ${metricName}, using default 0`); // Debug log
      }
    });

    console.log("Final data for radar chart:", lastValues); // Debug log

    // Normalize values to 0-1 scale
    const normalizedValues = {};

    // Define ideal ranges for each metric type
    const metricRanges = {
      // Accuracy: higher is better, typically 0-1
      accuracy: { min: 0, max: 1, higherIsBetter: true },
      // Precision: higher is better, typically 0-1
      precision: { min: 0, max: 1, higherIsBetter: true },
      // Fairness: higher is better (1 = perfect fairness), typically 0-1
      fairness: { min: 0, max: 1, higherIsBetter: true },
      // Bias: lower is better (0 = no bias), typically 0+
      bias: { min: 0, max: 1, higherIsBetter: false }
    };

    // Normalize each value based on its expected range
    Object.entries(lastValues).forEach(([metricName, value]) => {
      const range = metricRanges[metricName];
      if (!range) {
        // Default normalization if no specific range defined
        normalizedValues[metricName] = Math.min(Math.max(value, 0), 1);
        return;
      }

      // Get min/max for this metric
      const { min, max, higherIsBetter } = range;

      // Normalize to 0-1 range
      let normalizedValue = (value - min) / (max - min);

      // Clamp to 0-1
      normalizedValue = Math.min(Math.max(normalizedValue, 0), 1);

      // Invert if lower is better (so 1 is always best on the chart)
      if (!higherIsBetter) {
        normalizedValue = 1 - normalizedValue;
      }

      normalizedValues[metricName] = normalizedValue;
    });

    console.log("Normalized values:", normalizedValues); // Debug log

    // Format labels for better display
    const formattedLabels = Object.keys(normalizedValues).map(key =>
      key.charAt(0).toUpperCase() + key.slice(1)
    );

    const chartData = {
      labels: formattedLabels,
      datasets: [{
        label: 'Model Performance',
        data: Object.values(normalizedValues),
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 1,
        pointBackgroundColor: 'rgba(255, 99, 132, 1)',
        pointBorderColor: '#fff',
        pointHoverBackgroundColor: '#fff',
        pointHoverBorderColor: 'rgba(255, 99, 132, 1)',
      }]
    };

    // Special options for radar chart
    const radarOptions = {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          beginAtZero: true,
          min: 0,
          max: 1, // Fixed max since we normalized all values to 0-1
          ticks: {
            stepSize: 0.2,
            backdropColor: 'transparent'
          },
          grid: {
            circular: true
          },
          angleLines: {
            display: true
          },
          pointLabels: {
            font: {
              size: 14,
              weight: 'bold'
            }
          }
        }
      },
      plugins: {
        legend: {
          position: 'top',
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              // Show both normalized and original values in tooltip
              const metricName = formattedLabels[context.dataIndex].toLowerCase();
              const originalValue = lastValues[metricName];
              return [
                `Normalized: ${context.raw.toFixed(2)}`,
                `Original: ${originalValue.toFixed(3)}`
              ];
            }
          }
        }
      }
    };

    // Force a minimum of 3 data points for radar chart to render properly
    if (formattedLabels.length < 3) {
      console.log("Not enough data points for radar chart, need at least 3"); // Debug log
      return (
        <Box sx={{ mb: 4, mt: 2 }}>
          <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
            Multi-Metric Radar
          </Typography>
          <Typography>
            Waiting for more metrics data to display radar chart (need at least 3 metrics)
          </Typography>
        </Box>
      );
    }

    return (
      <Box sx={{ mb: 4, mt: 2 }}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
          Multi-Metric Radar
        </Typography>
        <Box sx={{ height: 400, width: '100%', maxWidth: '600px', margin: '0 auto' }}>
          <Radar data={chartData} options={radarOptions} />
        </Box>
        <Box sx={{ mt: 2, textAlign: 'center' }}>
          <Typography variant="caption" color="text.secondary">
            All metrics normalized to 0-1 scale where 1 is optimal performance
          </Typography>
        </Box>
      </Box>
    );
  };

  const renderMetricCharts = (metrics) => {
    return Object.entries(metrics).map(([metricKey, data]) => {
      if (!Array.isArray(data) || data.length === 0) {
        return (
          <Box key={metricKey} sx={{ p: 2 }}>
            <Typography>{metricKey}</Typography>
            <Typography>No data</Typography>
          </Box>
        );
      }

      const chartData = {
        labels: data.map((pt) => pt.step),
        datasets: [{
          label: metricKey,
          data: data.map((pt) => pt.value),
          borderColor: "#8884d8",
          backgroundColor: "#8884d8",
        }],
      };

      return (
        <Box key={metricKey} sx={{ mb: 3 }}>
          <Typography fontWeight="bold" sx={{ mb: 1 }}>
            {metricKey}
          </Typography>
          <Box sx={{ width: "100%", height: 300 }}>
            <Line data={chartData} options={getChartOptions(metricKey)} />
          </Box>
        </Box>
      );
    });
  };

  return (
    <Box sx={{ width: '100%' }}>
      {/* Always show radar chart at the top */}
      {renderRadarChart()}

      {/* Add training visualization */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h6" fontWeight="bold" sx={{ mb: 2 }}>
          Training Data Visualization
        </Typography>
        <TrainingVisualization data={visualizationData} />
      </Box>

      <Tabs
        value={currentTab}
        onChange={handleTabChange}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}
      >
        {groups.map((group, index) => (
          <Tab
            key={group}
            label={group.charAt(0).toUpperCase() + group.slice(1)}
          />
        ))}
      </Tabs>

      {groups.map((group, index) => (
        <TabPanel key={group} value={currentTab} index={index}>
          {renderMetricCharts(scalars[group])}
        </TabPanel>
      ))}
    </Box>
  );
}

// Helper component for tabs
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`metrics-tabpanel-${index}`}
      aria-labelledby={`metrics-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

/**
 * 5) Single-Page layout.
 *    - Top bar for brand & dark mode toggle
 *    - Fluid container (maxWidth="xl") for adaptive full-screen
 *    - 3 Paper sections stacked vertically: metrics, configs, logs
 */
export default function MainApp() {
  const [darkMode, setDarkMode] = useState(true);
  const toggleDarkMode = () => setDarkMode((prev) => !prev);

  const theme = createTheme({
    palette: {
      mode: darkMode ? "dark" : "light",
    },
    typography: {
      fontFamily: `"Inter", "Roboto", "Helvetica", "Arial", sans-serif`,
      fontSize: 14,
    },
  });

  // Data from the custom hook
  const {
    scalars,
    trainingState,
    currentConfig,
    inputConfig,
    isChanging,
    error,
    connectionState,
    isPending,
    handleInputChange,
    applyChanges,
    getChartOptions,
    handleStop,
    handleStart,
  } = useDashboardData();

  const renderStateValue = (value) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'boolean') return value.toString();
    if (typeof value === 'number') return value.toFixed(6);
    if (typeof value === 'string') return value;
    if (typeof value === 'object') return JSON.stringify(value);
    return value.toString();
  };

  const logsContainerRef = useRef(null);

  const scrollToBottom = () => {
    if (logsContainerRef.current) {
      logsContainerRef.current.scrollTop = logsContainerRef.current.scrollHeight;
    }
  };

  // Scroll to bottom when logs update
  useEffect(() => {
    if (trainingState?.stdout) {
      scrollToBottom();
    }
  }, [trainingState?.stdout]);

  // If data not loaded => spinner
  if (!trainingState) {
    return (
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <Typography sx={{ mt: 8, ml: 2 }}>Loading...</Typography>
      </ThemeProvider>
    );
  }

  // Overlay for Start/Stop
  const stateRunning = trainingState.running;
  const configRunning = inputConfig.running;
  let overlayText = "";
  if (isChanging) {
    if (configRunning && !stateRunning) overlayText = "Starting up…";
    if (!configRunning && stateRunning) overlayText = "Shutting down…";
  }
  const trainingStopped = !stateRunning && !configRunning && !isChanging;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />

      {/* Top Bar */}
      <AppBar position="fixed">
        <Toolbar>
          <Typography variant="h5" sx={{ flexGrow: 1, fontWeight: "bold" }}>
            Fairness Training Dashboard
          </Typography>
          <IconButton onClick={toggleDarkMode} sx={{ color: "inherit" }}>
            {darkMode ? <LightModeIcon /> : <DarkModeIcon />}
          </IconButton>
        </Toolbar>
      </AppBar>

      {/* Container => fluid & responsive up to 'xl' screens */}
      <Container
        maxWidth={false}
        sx={{
          mt: 12,
          minHeight: 'calc(100vh - 64px - 32px)',
          position: "relative",
          pb: 4,
          px: { xs: 3, sm: 4, md: 6 }
        }}
      >
        {/* Show errors */}
        {error && (
          <Typography color="error" sx={{ mb: 2 }}>
            {error}
          </Typography>
        )}

        {/* Pending Changes Indicator */}
        {isPending() && (
          <Paper
            sx={{
              p: 2,
              mb: 2,
              backgroundColor: 'warning.main',
              color: 'warning.contrastText',
              display: 'flex',
              alignItems: 'center',
              gap: 2
            }}
          >
            <CircularProgress
              size={20}
              thickness={5}
              sx={{ color: 'warning.contrastText' }}
            />
            <Typography>
              Changes pending... Waiting for trainer to update
            </Typography>
          </Paper>
        )}

        {/* Start/Stop overlay */}
        {overlayText && (
          <Box
            sx={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              bgColor: "rgba(0,0,0,0.3)",
              zIndex: 9999,
              color: "white",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 32,
              fontWeight: "bold",
            }}
          >
            {overlayText}
          </Box>
        )}

        {/* Buffer space */}
        <Box sx={{ mb: 4 }} />

        <Box sx={{
          display: "flex",
          gap: 2,
          minHeight: '100%',
          pb: 4,
          mx: 'auto',
          maxWidth: '1800px',
        }}>
          {/* Left side: Logs and Metrics */}
          <Box sx={{
            flex: '1 1 auto',
            overflow: 'auto',
            minWidth: '0',
            maxWidth: '70%',  // Increased to 70%
            width: '70%',     // Set preferred width
            mr: 2
          }}>
            {/* Training Logs - Now First */}
            <Paper sx={{ p: 2, mb: 2 }} elevation={2}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Training Logs
              </Typography>
              <Box
                ref={logsContainerRef}
                sx={{
                  backgroundColor: 'black',
                  color: 'lightgreen',
                  p: 2,
                  borderRadius: 1,
                  fontFamily: 'monospace',
                  maxHeight: '400px',
                  overflowY: 'auto',
                  overflowX: 'auto',
                  whiteSpace: 'pre-wrap',
                  width: '100%',
                  minWidth: '500px',
                }}
              >
                {trainingState?.stdout || 'No logs yet.'}
              </Box>
            </Paper>

            {/* Metrics Paper */}
            <Paper sx={{ p: 2 }} elevation={2}>
              <Typography variant="h6" fontWeight="bold" gutterBottom>
                Metrics
              </Typography>
              {Object.keys(scalars).length === 0 ? (
                <Box sx={{ p: 2 }}>
                  <CircularProgress />
                </Box>
              ) : (
                <MetricsTabs scalars={scalars} getChartOptions={getChartOptions} trainingState={trainingState} />
              )}
            </Paper>
          </Box>

          {/* Right side: Config Panel */}
          <Box sx={{
            width: { xs: '350px', md: '400px', lg: '450px' },  // Responsive width
            flexShrink: 0,
            position: 'sticky',
            alignSelf: 'flex-start',
            ml: 2,
            mr: { xs: 2, sm: 3, md: 4 }
          }}>
            <Paper
              sx={{
                p: 2,
                height: 'auto',
                position: 'relative',
                display: 'flex',
                flexDirection: 'column'
              }}
              elevation={2}
            >
              <Box sx={{ position: 'relative', zIndex: 0 }}>
                <Typography variant="h6" fontWeight="bold" gutterBottom>
                  Configs
                </Typography>

                {/* Current Config as table - only show non-metric settings */}
                <TableContainer sx={{ maxHeight: 'none' }}>
                  <Table size="small">
                    <TableHead>
                      <TableRow>
                        <TableCell colSpan={2} align="center">
                          Current Settings
                        </TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {trainingState && Object.entries(trainingState)
                        .filter(([key]) =>
                          // Only show non-metric settings
                          !key.includes('_acc') &&
                          !key.includes('_md') &&
                          !key.includes('_dp') &&
                          !key.includes('_eo') &&
                          !key.includes('_auc') &&
                          key !== 'stdout' &&
                          key !== 'metrics'
                        )
                        .sort(([keyA], [keyB]) => keyA.localeCompare(keyB))
                        .map(([key, value]) => (
                          <TableRow key={key}>
                            <TableCell
                              sx={{
                                textTransform: 'capitalize',
                                fontWeight: 'medium'
                              }}
                            >
                              {key.replace(/_/g, ' ')}
                            </TableCell>
                            <TableCell>{renderStateValue(value)}</TableCell>
                          </TableRow>
                        ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                <Divider sx={{ my: 2 }} />

                {/* Config Controls - disabled when pending */}
                <Box sx={{
                  opacity: isPending() ? 0.5 : 1,
                  pointerEvents: isPending() ? 'none' : 'auto'
                }}>
                  <Box sx={{ mb: 1 }}>
                    Running: {String(inputConfig.running)}
                  </Box>
                  {trainingState?.running ? (
                    <Button
                      variant="outlined"
                      sx={{ mb: 2 }}
                      onClick={handleStop}
                      disabled={isPending()}
                    >
                      Stop
                    </Button>
                  ) : (
                    <Button
                      variant="outlined"
                      sx={{ mb: 2 }}
                      onClick={handleStart}
                      disabled={isPending()}
                    >
                      Start
                    </Button>
                  )}
                  <TextField
                    label="Batch Size"
                    type="number"
                    fullWidth
                    margin="dense"
                    value={inputConfig.batch_size}
                    onChange={(e) => handleInputChange("batch_size", e.target.value)}
                    disabled={isPending()}
                    sx={{ mb: 1 }}
                  />
                  <TextField
                    label="Learning Rate"
                    type="number"
                    fullWidth
                    margin="dense"
                    value={inputConfig.learning_rate}
                    onChange={(e) => handleInputChange("learning_rate", e.target.value)}
                    disabled={isPending()}
                  />
                  <Button
                    variant="contained"
                    sx={{ mt: 2 }}
                    onClick={applyChanges}
                    disabled={isPending()}
                  >
                    Apply
                  </Button>
                </Box>
              </Box>

              {/* Pending Changes Overlay */}
              {isPending() && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1,
                    backdropFilter: 'blur(2px)',
                  }}
                >
                  <Box sx={{ textAlign: 'center' }}>
                    <CircularProgress sx={{ mb: 2 }} />
                    <Typography
                      variant="h6"
                      sx={{
                        color: 'white',
                        fontWeight: 'bold',
                        animation: 'bounce 1s infinite',
                        '@keyframes bounce': {
                          '0%, 100%': { transform: 'translateY(0)' },
                          '50%': { transform: 'translateY(-10px)' },
                        },
                      }}
                    >
                      Changes Pending...
                    </Typography>
                  </Box>
                </Box>
              )}

              {/* Training Stopped Overlay - only show when not pending */}
              {trainingStopped && !isPending() && (
                <Box
                  sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    backgroundColor: 'rgba(0, 0, 0, 0.7)',
                    display: 'flex',
                    flexDirection: 'column',
                    alignItems: 'center',
                    justifyContent: 'center',
                    gap: 3,
                    zIndex: 1,
                    backdropFilter: 'blur(2px)',
                  }}
                >
                  <Typography
                    variant="h5"
                    sx={{
                      color: 'white',
                      fontWeight: 'bold',
                      textAlign: 'center',
                      animation: 'bounce 1s infinite',
                      '@keyframes bounce': {
                        '0%, 100%': { transform: 'translateY(0)' },
                        '50%': { transform: 'translateY(-10px)' },
                      },
                    }}
                  >
                    Training Stopped
                  </Typography>
                  <Button
                    variant="contained"
                    size="large"
                    onClick={handleStart}
                    sx={{
                      fontSize: '1.2rem',
                      px: 4,
                      py: 1.5,
                      backgroundColor: 'primary.main',
                      '&:hover': {
                        backgroundColor: 'primary.dark',
                      },
                      position: 'relative',
                    }}
                  >
                    Start Training
                  </Button>
                </Box>
              )}
            </Paper>
          </Box>
        </Box>
      </Container>

      {/* Single Connection/Loading Modal */}
      <Modal
        open={!connectionState.isConnected || !trainingState}
        aria-labelledby="connection-modal-title"
        aria-describedby="connection-modal-description"
      >
        <Box sx={{
          position: 'absolute',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -50%)',
          width: 400,
          bgcolor: 'background.paper',
          borderRadius: 2,
          boxShadow: 24,
          p: 4,
        }}>
          <Typography id="connection-modal-title" variant="h6" component="h2" gutterBottom>
            Connection Error
          </Typography>
          <Typography id="connection-modal-description" sx={{ mt: 2, mb: 3 }}>
            {connectionState.error || 'Connecting to server...'}
          </Typography>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <CircularProgress size={20} />
            <Typography variant="body2" color="text.secondary">
              Reconnect attempt: {connectionState.reconnectAttempt}
            </Typography>
          </Box>
        </Box>
      </Modal>
    </ThemeProvider>
  );
}