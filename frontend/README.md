# Skyview Search - Desktop Frontend

A WPF desktop application for semantic image search, providing an intuitive interface to search aerial landscape imagery using natural language queries.

## Overview

This frontend connects to the Skyview Search API backend and provides:
- Natural language image search
- Thumbnail grid display of results
- Full image viewer
- Real-time connection status
- Category and score display for each result

## Screenshots
<img width="1917" height="1018" alt="Screenshot 2026-01-09 230633" src="https://github.com/user-attachments/assets/9d616b25-be92-4922-aa26-a32e45751dc4" />

## Requirements

- **Windows 10/11**
- **.NET 8.0 SDK** or later
- **Skyview Search Backend** running on `http://127.0.0.1:8000`

## Installation

### 1. Install .NET SDK

Download and install from: https://dotnet.microsoft.com/download/dotnet/8.0

Verify installation:
```cmd
dotnet --version
```

### 2. Clone/Download the Project

```cmd
cd C:\Users\YourUsername\Documents\skyview-search\frontend
```

### 3. Restore Dependencies

```cmd
cd SkyviewSearch
dotnet restore
```

### 4. Build

```cmd
dotnet build
```

## Running the Application

### Step 1: Start the Backend

Open a terminal and run:
```cmd
cd backend
.\.venv\Scripts\activate
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Step 2: Start the Frontend

Open another terminal:
```cmd
cd frontend\SkyviewSearch
dotnet run
```

Or run the compiled executable:
```cmd
.\bin\Debug\net8.0-windows\SkyviewSearch.exe
```

## Usage

### Search for Images

1. Type a search query in the search bar (e.g., "airport runway", "mountain peaks")
2. Press **Enter** or click the **Search** button
3. Results appear as a grid of thumbnails

### Adjust Result Count

Use the dropdown next to the search bar to select how many results to display (5, 10, 20, or 30).

### Clear Results

Click the **Clear** button to reset the search and clear all results.

## Project Structure

```
frontend/
├── SkyviewSearch.sln                 # Visual Studio solution
└── SkyviewSearch/
    ├── SkyviewSearch.csproj          # Project configuration
    ├── App.xaml                      # Application resources & styles
    ├── App.xaml.cs                   # Application entry point
    │
    ├── Models/                       # Data models
    │   ├── SearchResult.cs           # Single search result
    │   ├── SearchResponse.cs         # API search response
    │   └── StatsResponse.cs          # API stats response
    │
    ├── Services/                     # Backend communication
    │   └── ApiService.cs             # HTTP client for API calls
    │
    ├── ViewModels/                   # MVVM ViewModels
    │   ├── MainViewModel.cs          # Main window logic
    │   └── ImageItem.cs              # Image display item
    │
    ├── Views/                        # UI windows
    │   ├── MainWindow.xaml           # Main application window
    │   ├── MainWindow.xaml.cs        # Main window code-behind
    │   ├── ImageViewerWindow.xaml    # Full image viewer
    │   └── ImageViewerWindow.xaml.cs # Image viewer code-behind
    │
    └── Converters/                   # XAML value converters
        └── Converters.cs             # Bool/visibility converters
```

## Architecture

The application follows the **MVVM (Model-View-ViewModel)** pattern:

```
┌─────────────────────────────────────────────────────────┐
│                        View                             │
│  ┌─────────────────┐    ┌─────────────────────────┐    │
│  │  MainWindow     │    │  ImageViewerWindow      │    │
│  │  (XAML + C#)    │    │  (XAML + C#)            │    │
│  └────────┬────────┘    └─────────────────────────┘    │
│           │                                             │
│           ▼                                             │
│  ┌─────────────────────────────────────────────┐       │
│  │              ViewModel                       │       │
│  │  ┌─────────────────┐  ┌─────────────────┐   │       │
│  │  │ MainViewModel   │  │ ImageItem       │   │       │
│  │  │ - SearchQuery   │  │ - Thumbnail     │   │       │
│  │  │ - SearchResults │  │ - Caption       │   │       │
│  │  │ - StatusMessage │  │ - Category      │   │       │
│  │  └────────┬────────┘  └─────────────────┘   │       │
│  └───────────┼─────────────────────────────────┘       │
│              │                                          │
│              ▼                                          │
│  ┌─────────────────────────────────────────────┐       │
│  │              Services                        │       │
│  │  ┌─────────────────────────────────────┐    │       │
│  │  │ ApiService                          │    │       │
│  │  │ - SearchAsync()                     │    │       │
│  │  │ - GetThumbnailAsync()               │    │       │
│  │  │ - GetImageAsync()                   │    │       │
│  │  └─────────────────────────────────────┘    │       │
│  └─────────────────────────────────────────────┘       │
│                         │                               │
└─────────────────────────┼───────────────────────────────┘
                          │ HTTP
                          ▼
              ┌───────────────────────┐
              │   Backend API         │
              │   localhost:8000      │
              └───────────────────────┘
```

## API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check |
| `/search?q={query}&k={count}` | GET | Search images |
| `/thumbnail/{uuid}?size={size}` | GET | Get thumbnail |
| `/image/{uuid}` | GET | Get full image |
| `/stats` | GET | Get database statistics |

## Configuration

### "Connection refused" or "Disconnected" status

1. Ensure backend is running on port 8000
2. Check firewall settings
3. Verify URL in ApiService.cs

### Images not loading

1. Check backend console for errors
2. Verify image files exist at the paths in database
3. Check thumbnail endpoint: `http://127.0.0.1:8000/thumbnail/{uuid}`

### Publish as Self-Contained

```cmd
dotnet publish -c Release -r win-x64 --self-contained true
```

This creates a standalone executable that doesn't require .NET runtime installed.

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Newtonsoft.Json | 13.0.3 | JSON serialization |

## Future Improvements

- [ ] Image upload functionality
- [ ] Search history
- [ ] Favorites/bookmarks
- [ ] Export results
- [ ] Dark mode
- [ ] Keyboard shortcuts
- [ ] Batch operations

## Author

Pinak Ganatra
