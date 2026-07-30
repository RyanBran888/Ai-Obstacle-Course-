using Godot;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.IO;
using System.Diagnostics;
using System.Linq;

public partial class Node2d : Node2D
{
	private bool _redStartPlaced = false;
	private bool _blueStartPlaced = false;
	private bool _goalPlaced = false;
	private List<Dictionary<string, object>> _objects = new();
	private TileType _currentType;
	private Vector2I[,] _occupiedSize = new Vector2I[GridWidth, GridHeight];
	private MenuButton _optionButton;
	private Node2D _placedObjects;
	private int[,] _gameState = new int[GridWidth, GridHeight];
	private Vector2I _currentSize = new Vector2I(1, 1);
	private PackedScene[] _objectScenes;
	private const int CellSize = 32;
	private const int GridWidth = 20;
	private const int GridHeight = 20;

	// -- AI run/replay state --
	private string pythonExecutable = "python";
	private InferenceResult _lastResult;
	private int _replayIndex = 0;
	private bool _replaying = false;
	private double _replayTimer = 0.0;
	private const double ReplayStepSeconds = 0.15; // animation speed, tweak freely
	private Node2D _redAgentNode;   // captured in PlaceObject() when RedStart placed
	private Node2D _blueAgentNode;  // captured in PlaceObject() when BlueStart placed
	private List<Node2D> _keyNodes = new();   // captured in placement order
	private List<Node2D> _doorNodes = new();  // captured in placement order
	private List<Node2D> _ballNodes = new();  // captured in placement order

	private Vector2I[] _objectSizes =
	{
		new Vector2I(1,1), // RKey
		new Vector2I(1,1), // BKey
		new Vector2I(1,1), // RDoor
		new Vector2I(1,1), // BDoor
		new Vector2I(5,1), // SBall
		new Vector2I(11,1), // BBall
		new Vector2I(1,1),  // Wall
		new Vector2I(1,1), //lava
		new Vector2I(1,1), //end
		new Vector2I(1,1), //RStart
		new Vector2I(1,1) //BStart
	};
	private Vector2 GridOrigin = new Vector2(482.527f, 29.382f);

	private Node2D[,] _occupied = new Node2D[GridWidth, GridHeight];
	private Area2D _draggingObject;
	private Area2D[] _templates;
	private int _currentGridX;
	private int _currentGridY;

	public override void _Ready()
	{
		_templates = new Area2D[]
		{
			GetNode<Area2D>("Templates/RKey"),
			GetNode<Area2D>("Templates/BKey"),
			GetNode<Area2D>("Templates/RDoor"),
			GetNode<Area2D>("Templates/BDoor"),
			GetNode<Area2D>("Templates/SBall"),
			GetNode<Area2D>("Templates/BBall"),
			GetNode<Area2D>("Templates/Wall"),
			GetNode<Area2D>("Templates/Lava"),
			GetNode<Area2D>("Templates/End"),
			GetNode<Area2D>("Templates/RStart"),
			GetNode<Area2D>("Templates/BStart")
		};

		_optionButton = GetNode<MenuButton>("Control/MenuButton");

		_optionButton.GetPopup().IdPressed += OnItemSelected;
		_placedObjects = GetNode<Node2D>("PlacedObjects");
	}

	private void SaveGameStateJson()
	{
		List<List<int>> grid = new();

		for (int y = 0; y < GridHeight; y++)
		{
			List<int> row = new();

			for (int x = 0; x < GridWidth; x++)
			{
				row.Add(_gameState[x, y]);
			}

			grid.Add(row);
		}

		var levelData = new
		{
			version = 1,

			width = GridWidth,
			height = GridHeight,

			objects = _objects,

			grid = grid
		};

		string json = JsonSerializer.Serialize(
			levelData,
			new JsonSerializerOptions
			{
				WriteIndented = true
			});

		// Saves next to your project.godot file
	string exeDir = System.IO.Path.GetDirectoryName(OS.GetExecutablePath());
	string path = System.IO.Path.Combine(exeDir, "AI", "level.json");
	File.WriteAllText(path, json);
	GD.Print("Saved JSON to: " + path);
	}

	public enum TileType
	{
		Empty = 0,
		Wall = 1,
		RedDoor = 2,
		BlueDoor = 3,
		RedKey = 4,
		BlueKey = 5,
		SmallBall = 6,
		BigBall = 7,
		Lava = 8,
		Goal = 9,
		RedStart = 10,
		BlueStart = 11
	}

	public override void _Process(double delta)
	{
		if (_replaying) { AdvanceReplay(delta); }

		if (_draggingObject == null)
			return;

		Vector2 mouse = GetGlobalMousePosition();

		int x = Mathf.FloorToInt((mouse.X - GridOrigin.X) / CellSize);
		int y = Mathf.FloorToInt((mouse.Y - GridOrigin.Y) / CellSize);

		_currentGridX = x;
		_currentGridY = y;

		int halfWidth = (_currentSize.X - 1) / 2;
		int halfHeight = (_currentSize.Y - 1) / 2;

		if (x - halfWidth >= 0 &&
			x + halfWidth < GridWidth &&
			y - halfHeight >= 0 &&
			y + halfHeight < GridHeight)
		{
			_draggingObject.Visible = true;

			_draggingObject.Position = new Vector2(
				GridOrigin.X + x * CellSize + CellSize / 2,
				GridOrigin.Y + y * CellSize + CellSize / 2
			);
		}
		else
		{
			_draggingObject.Visible = false;
		}
	}

	private bool CanPlaceObject(int centerX, int centerY)
	{
		int halfWidth = (_currentSize.X - 1) / 2;
		int halfHeight = (_currentSize.Y - 1) / 2;

		for (int x = -halfWidth; x <= halfWidth; x++)
		{
			for (int y = -halfHeight; y <= halfHeight; y++)
			{
				int checkX = centerX + x;
				int checkY = centerY + y;

				if (checkX < 0 || checkX >= GridWidth ||
					checkY < 0 || checkY >= GridHeight)
				{
					return false;
				}

				if (_occupied[checkX, checkY] != null)
				{
					return false;
				}
			}
		}

		return true;
	}

	public override void _Draw()
	{
		Color gridColor = Colors.Gray;

		for (int x = 0; x <= GridWidth; x++)
		{
			float xpos = GridOrigin.X + x * CellSize;

			DrawLine(
				new Vector2(xpos, GridOrigin.Y),
				new Vector2(xpos, GridOrigin.Y + GridHeight * CellSize),
				gridColor
			);
		}

		for (int y = 0; y <= GridHeight; y++)
		{
			float ypos = GridOrigin.Y + y * CellSize;

			DrawLine(
				new Vector2(GridOrigin.X, ypos),
				new Vector2(GridOrigin.X + GridWidth * CellSize, ypos),
				gridColor
			);
		}
	}

	private void OnItemSelected(long index)
	{
		if (_draggingObject != null)
			_draggingObject.QueueFree();

		_draggingObject = (Area2D)_templates[index].Duplicate();

		_currentSize = _objectSizes[index];

		_placedObjects.AddChild(_draggingObject);
		_currentType = index switch
		{
			0 => TileType.RedKey,
			1 => TileType.BlueKey,
			2 => TileType.RedDoor,
			3 => TileType.BlueDoor,
			4 => TileType.SmallBall,
			5 => TileType.BigBall,
			6 => TileType.Wall,
			7 => TileType.Lava,
			8 => TileType.Goal,
			9 => TileType.RedStart,
			10 => TileType.BlueStart,
			_ => TileType.Empty
		};
	}

	public override void _Input(InputEvent e)
	{
		if (_draggingObject == null)
			return;

		if (e is InputEventMouseButton mouse &&
			mouse.ButtonIndex == MouseButton.Left &&
			mouse.Pressed)
		{
			PlaceObject();
		}
	}

	private void PlaceObject()
	{
		// Only allow one Goal
		if (_currentType == TileType.Goal && _goalPlaced)
		{
			GD.Print("Goal already exists.");
			return;
		}

		// Only allow one Red Start
		if (_currentType == TileType.RedStart && _redStartPlaced)
		{
			GD.Print("Red Start already exists.");
			return;
		}

		// Only allow one Blue Start
		if (_currentType == TileType.BlueStart && _blueStartPlaced)
		{
			GD.Print("Blue Start already exists.");
			return;
		}

		if (!CanPlaceObject(_currentGridX, _currentGridY))
		{
			GD.Print("Cannot place here");
			return;
		}

		GD.Print("Placed object!");

		// Mark every square this object takes up
		int halfWidth = (_currentSize.X - 1) / 2;
		int halfHeight = (_currentSize.Y - 1) / 2;

		for (int x = -halfWidth; x <= halfWidth; x++)
		{
			for (int y = -halfHeight; y <= halfHeight; y++)
			{
				int checkX = _currentGridX + x;
				int checkY = _currentGridY + y;

				_occupied[checkX, checkY] = _draggingObject;
				_gameState[checkX, checkY] = (int)_currentType;
				_occupiedSize[checkX, checkY] = _currentSize;
			}
		}

		_objects.Add(new Dictionary<string, object>
		{
			{ "type", _currentType.ToString() },
			{ "x", _currentGridX },
			{ "y", _currentGridY },
			{ "width", _currentSize.X },
			{ "height", _currentSize.Y }
		});

		switch (_currentType)
		{
			case TileType.Goal:
				_goalPlaced = true;
				break;

			case TileType.RedStart:
				_redStartPlaced = true;
				break;

			case TileType.BlueStart:
				_blueStartPlaced = true;
				break;
		}

		// Leave the placed object where it is
		_draggingObject.Modulate = Colors.White;

		// Capture references for the AI run/replay to control later
		if (_currentType == TileType.RedStart) _redAgentNode = _draggingObject;
		else if (_currentType == TileType.BlueStart) _blueAgentNode = _draggingObject;
		else if (_currentType == TileType.RedKey || _currentType == TileType.BlueKey) _keyNodes.Add(_draggingObject);
		else if (_currentType == TileType.RedDoor || _currentType == TileType.BlueDoor) _doorNodes.Add(_draggingObject);
		else if (_currentType == TileType.SmallBall)
			_ballNodes.Add(_draggingObject.GetNode<Node2D>("Ball/SBallBall"));
		else if (_currentType == TileType.BigBall)
			_ballNodes.Add(_draggingObject.GetNode<Node2D>("Ball/BBallBall"));
		
		// Create a new preview of the same object
		_draggingObject = (Area2D)_templates[_currentType switch
		{
			TileType.RedKey => 0,
			TileType.BlueKey => 1,
			TileType.RedDoor => 2,
			TileType.BlueDoor => 3,
			TileType.SmallBall => 4,
			TileType.BigBall => 5,
			TileType.Wall => 6,
			TileType.Lava => 7,
			TileType.Goal => 8,
			TileType.RedStart => 9,
			TileType.BlueStart => 10,
			_ => 0
		}].Duplicate();

		_placedObjects.AddChild(_draggingObject);
	}

	private List<int> GetGameState()
	{
		List<int> data = new();

		for (int y = 0; y < GridHeight; y++)
		{
			for (int x = 0; x < GridWidth; x++)
			{
				data.Add(_gameState[x, y]);
			}
		}

		return data;
	}

	private void PushyPushy()
	{
		RunAI();
	}

	// ==========================================================================
	// AI run + replay
	// ==========================================================================

	public class KeyState
	{
		public string color { get; set; }
		public List<int> pos { get; set; }
		public bool collected { get; set; }
	}

	public class DoorState
	{
		public string color { get; set; }
		public List<int> pos { get; set; }
		public bool open { get; set; }
	}

	public class BallState
	{
		public string color { get; set; }
		public List<double> center { get; set; }
	}

	public class TrajectoryStep
	{
		public int step { get; set; }
		public List<List<int>> positions { get; set; }
		public List<int> actions { get; set; }
		public List<string> events { get; set; }
		public List<KeyState> keys { get; set; }
		public List<DoorState> doors { get; set; }
		public List<BallState> balls { get; set; }
	}

	public class InferenceResult
	{
		public bool success { get; set; }
		public int steps { get; set; }
		public string failure_reason { get; set; }
		public List<TrajectoryStep> trajectory { get; set; }
		public string error { get; set; }
	}

	private void RunAI()
	{
		if (_redAgentNode == null || _blueAgentNode == null)
		{
			GD.PrintErr("Place both RedStart and BlueStart before running the AI.");
			return;
		}

		// 1. Write the current course to res://AI/level.json
		SaveGameStateJson();

		string exeDir = System.IO.Path.GetDirectoryName(OS.GetExecutablePath());
string aiDir = System.IO.Path.Combine(exeDir, "AI");
		string levelPath = System.IO.Path.Combine(aiDir, "level.json");
		string resultPath = System.IO.Path.Combine(aiDir, "result.json");
string inferExe = System.IO.Path.Combine(aiDir, "infer", "infer.exe");

var psi = new ProcessStartInfo
{
	FileName = inferExe,
	Arguments = $"--level \"{levelPath}\" --out \"{resultPath}\"",
	WorkingDirectory = aiDir,
	RedirectStandardOutput = true,
	RedirectStandardError = true,
	UseShellExecute = false,
	CreateNoWindow = true,
};

		try
		{
			using var process = Process.Start(psi);
			string stdout = process.StandardOutput.ReadToEnd();
			string stderr = process.StandardError.ReadToEnd();
			process.WaitForExit();

			if (process.ExitCode != 0)
			{
				GD.PrintErr($"infer.py failed (exit {process.ExitCode}): {stderr}");
				return;
			}

			if (!string.IsNullOrEmpty(stdout))
				GD.Print(stdout);
		}
		catch (System.Exception e)
		{
			GD.PrintErr($"Failed to launch Python: {e.Message}. "
				+ $"Check that '{pythonExecutable}' is on PATH and torch is installed.");
			return;
		}

		if (!File.Exists(resultPath))
		{
			GD.PrintErr("infer.py did not produce result.json");
			return;
		}

		string json = File.ReadAllText(resultPath);
		InferenceResult result;
		try
		{
			result = JsonSerializer.Deserialize<InferenceResult>(json);
		}
		catch (System.Exception e)
		{
			GD.PrintErr($"Could not parse result.json: {e.Message}");
			return;
		}

		if (result.error != null)
		{
			GD.PrintErr($"Inference error: {result.error}");
			return;
		}

		GD.Print($"Run complete: success={result.success}, steps={result.steps}"
			+ (result.failure_reason != null ? $", reason={result.failure_reason}" : ""));

		if (result.trajectory.Count > 0)
		{
			ApplyFrame(result.trajectory[0]);
		}

		_lastResult = result;
		_replayIndex = 1; // step 0 already applied above
		_replayTimer = 0.0;
		_replaying = true;
	}

	private void AdvanceReplay(double delta)
	{
		if (_lastResult == null || _replayIndex >= _lastResult.trajectory.Count)
		{
			_replaying = false;
			return;
		}

		_replayTimer += delta;
		if (_replayTimer < ReplayStepSeconds)
			return;
		_replayTimer = 0.0;

		var frame = _lastResult.trajectory[_replayIndex];
		ApplyFrame(frame);

		if (frame.events != null && frame.events.Count > 0)
			GD.Print($"step {frame.step}: {string.Join(", ", frame.events)}");

		_replayIndex++;
	}

	private void ApplyFrame(TrajectoryStep frame)
	{
		_redAgentNode.Position = GridToWorld(frame.positions[0][0], frame.positions[0][1]);
		_blueAgentNode.Position = GridToWorld(frame.positions[1][0], frame.positions[1][1]);

		if (frame.keys != null)
		{
			for (int i = 0; i < frame.keys.Count && i < _keyNodes.Count; i++)
				_keyNodes[i].Visible = !frame.keys[i].collected;
		}

		if (frame.doors != null)
		{
			for (int i = 0; i < frame.doors.Count && i < _doorNodes.Count; i++)
				_doorNodes[i].Visible = !frame.doors[i].open;
		}

		if (frame.balls != null)
		{
			for (int i = 0; i < frame.balls.Count && i < _ballNodes.Count; i++)
			{
				var c = frame.balls[i].center;
				_ballNodes[i].GlobalPosition = GridToWorld((float)c[0], (float)c[1]);
			}
		}
	}

	private Vector2 GridToWorld(float gx, float gy)
	{
		return new Vector2(
			GridOrigin.X + gx * CellSize + CellSize / 2,
			GridOrigin.Y + gy * CellSize + CellSize / 2
		);
	}
}
