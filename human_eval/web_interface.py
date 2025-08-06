"""
Web Interface for Human Evaluation System
FastAPI-based web interface for human evaluators to review AI responses.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
from datetime import datetime

from .evaluation_manager import HumanEvaluationManager, EvaluationTask, HumanEvaluator


class WebEvaluationInterface:
    """
    Web-based interface for human evaluation of AI responses.
    Provides a user-friendly interface for evaluators to review and score responses.
    """
    
    def __init__(self, evaluation_manager: HumanEvaluationManager, config: Optional[Dict] = None):
        self.evaluation_manager = evaluation_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # FastAPI app setup
        self.app = FastAPI(
            title="ShadowBench Human Evaluation Interface",
            description="Web interface for human evaluation of AI responses",
            version="1.0.0"
        )
        
        # Setup templates and static files
        self.templates_dir = Path(__file__).parent / "templates"
        self.static_dir = Path(__file__).parent / "static"
        
        # Create directories if they don't exist
        self.templates_dir.mkdir(exist_ok=True)
        self.static_dir.mkdir(exist_ok=True)
        
        self.templates = Jinja2Templates(directory=str(self.templates_dir))
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory=str(self.static_dir)), name="static")
        
        # Active sessions
        self.active_sessions: Dict[str, Dict] = {}
        
        # Setup routes
        self._setup_routes()
        
        # Create default templates
        self._create_default_templates()
        self._create_default_static_files()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def home(request: Request):
            """Home page with evaluation overview."""
            stats = self.evaluation_manager.get_evaluation_statistics()
            return self.templates.TemplateResponse("home.html", {
                "request": request,
                "stats": stats
            })
        
        @self.app.get("/register", response_class=HTMLResponse)
        async def register_page(request: Request):
            """Evaluator registration page."""
            return self.templates.TemplateResponse("register.html", {"request": request})
        
        @self.app.post("/register")
        async def register_evaluator(
            request: Request,
            name: str = Form(...),
            email: str = Form(...),
            expertise: str = Form(...)
        ):
            """Register a new evaluator."""
            expertise_areas = [area.strip() for area in expertise.split(",")]
            evaluator_id = self.evaluation_manager.register_evaluator(name, email, expertise_areas)
            
            # Create session
            self.active_sessions[evaluator_id] = {
                "evaluator_id": evaluator_id,
                "name": name,
                "login_time": datetime.now()
            }
            
            return JSONResponse({
                "success": True,
                "evaluator_id": evaluator_id,
                "message": f"Welcome {name}! You have been registered as an evaluator."
            })
        
        @self.app.get("/dashboard/{evaluator_id}", response_class=HTMLResponse)
        async def evaluator_dashboard(request: Request, evaluator_id: str):
            """Evaluator dashboard."""
            if evaluator_id not in self.evaluation_manager.evaluators:
                raise HTTPException(status_code=404, detail="Evaluator not found")
            
            evaluator = self.evaluation_manager.evaluators[evaluator_id]
            assigned_tasks = self.evaluation_manager.get_assigned_tasks(evaluator_id)
            pending_tasks = self.evaluation_manager.get_pending_tasks(evaluator_id)[:10]  # Show top 10
            
            return self.templates.TemplateResponse("dashboard.html", {
                "request": request,
                "evaluator": evaluator,
                "assigned_tasks": assigned_tasks,
                "pending_tasks": pending_tasks,
                "evaluation_criteria": self.evaluation_manager.evaluation_criteria
            })
        
        @self.app.get("/evaluate/{task_id}", response_class=HTMLResponse)
        async def evaluation_page(request: Request, task_id: str, evaluator_id: str = None):
            """Individual task evaluation page."""
            if task_id not in self.evaluation_manager.tasks:
                raise HTTPException(status_code=404, detail="Task not found")
            
            task = self.evaluation_manager.tasks[task_id]
            
            # Check if evaluator can access this task
            if evaluator_id and task.assigned_to != evaluator_id:
                # Try to assign task to evaluator
                if task.status == 'pending':
                    assigned_task = self.evaluation_manager.assign_task(evaluator_id, task_id)
                    if not assigned_task:
                        raise HTTPException(status_code=403, detail="Cannot assign task to evaluator")
                else:
                    raise HTTPException(status_code=403, detail="Task not accessible")
            
            return self.templates.TemplateResponse("evaluate.html", {
                "request": request,
                "task": task,
                "evaluation_criteria": self.evaluation_manager.evaluation_criteria,
                "evaluator_id": evaluator_id
            })
        
        @self.app.post("/submit_evaluation")
        async def submit_evaluation(
            request: Request,
            task_id: str = Form(...),
            evaluator_id: str = Form(...),
            harmfulness: int = Form(...),
            accuracy: int = Form(...),
            helpfulness: int = Form(...),
            coherence: int = Form(...),
            appropriateness: int = Form(...),
            comments: str = Form("")
        ):
            """Submit evaluation results."""
            evaluation_data = {
                "harmfulness": harmfulness,
                "accuracy": accuracy,
                "helpfulness": helpfulness,
                "coherence": coherence,
                "appropriateness": appropriateness,
                "comments": comments,
                "submitted_at": datetime.now().isoformat()
            }
            
            success = self.evaluation_manager.submit_evaluation(task_id, evaluator_id, evaluation_data)
            
            if success:
                return JSONResponse({
                    "success": True,
                    "message": "Evaluation submitted successfully!"
                })
            else:
                return JSONResponse({
                    "success": False,
                    "message": "Failed to submit evaluation."
                })
        
        @self.app.get("/api/tasks/pending")
        async def get_pending_tasks(evaluator_id: Optional[str] = None):
            """API endpoint for pending tasks."""
            tasks = self.evaluation_manager.get_pending_tasks(evaluator_id)
            return [{"task_id": task.task_id, "scenario": task.scenario_name, 
                    "model": task.model_name, "created_at": task.created_at.isoformat()}
                   for task in tasks]
        
        @self.app.get("/api/statistics")
        async def get_statistics():
            """API endpoint for evaluation statistics."""
            return self.evaluation_manager.get_evaluation_statistics()
        
        @self.app.get("/consensus/{scenario_name}/{model_name}")
        async def consensus_analysis(request: Request, scenario_name: str, model_name: str):
            """Consensus analysis page."""
            consensus_data = self.evaluation_manager.calculate_consensus(scenario_name, model_name)
            
            return self.templates.TemplateResponse("consensus.html", {
                "request": request,
                "consensus_data": consensus_data,
                "scenario_name": scenario_name,
                "model_name": model_name
            })
        
        @self.app.get("/ab_test/{test_id}")
        async def ab_test_results(request: Request, test_id: str):
            """A/B test results page."""
            analysis = self.evaluation_manager.analyze_ab_test(test_id)
            
            if "error" in analysis:
                raise HTTPException(status_code=404, detail=analysis["error"])
            
            return self.templates.TemplateResponse("ab_test.html", {
                "request": request,
                "analysis": analysis
            })
        
        @self.app.post("/api/assign_task")
        async def assign_task(evaluator_id: str, task_id: Optional[str] = None):
            """API endpoint to assign a task."""
            task = self.evaluation_manager.assign_task(evaluator_id, task_id)
            
            if task:
                return {"success": True, "task_id": task.task_id}
            else:
                return {"success": False, "message": "No suitable task found"}
    
    def _create_default_templates(self):
        """Create default HTML templates."""
        
        # Base template
        base_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}ShadowBench Human Evaluation{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/custom.css" rel="stylesheet">
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="/">ShadowBench Evaluation</a>
            <div class="navbar-nav">
                <a class="nav-link" href="/">Home</a>
                <a class="nav-link" href="/register">Register</a>
            </div>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    <script src="/static/app.js"></script>
</body>
</html>
        """
        
        # Home template
        home_template = """
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <h1>ShadowBench Human Evaluation Interface</h1>
        <p class="lead">Welcome to the human evaluation interface for AI safety research.</p>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Evaluation Statistics</h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Tasks Created:</strong> {{ stats.basic_stats.tasks_created }}</p>
                        <p><strong>Tasks Completed:</strong> {{ stats.basic_stats.tasks_completed }}</p>
                        <p><strong>Completion Rate:</strong> {{ "%.1f"|format(stats.completion_rate * 100) }}%</p>
                        <p><strong>Active Evaluators:</strong> {{ stats.basic_stats.evaluators_active }}</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">
                        <h5>Quick Actions</h5>
                    </div>
                    <div class="card-body">
                        <a href="/register" class="btn btn-primary btn-block mb-2">Register as Evaluator</a>
                        <a href="/api/statistics" class="btn btn-secondary btn-block">View Full Statistics</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        """
        
        # Register template
        register_template = """
{% extends "base.html" %}

{% block content %}
<div class="row justify-content-center">
    <div class="col-md-6">
        <div class="card">
            <div class="card-header">
                <h3>Evaluator Registration</h3>
            </div>
            <div class="card-body">
                <form id="registrationForm">
                    <div class="mb-3">
                        <label for="name" class="form-label">Full Name</label>
                        <input type="text" class="form-control" id="name" name="name" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="email" class="form-label">Email Address</label>
                        <input type="email" class="form-control" id="email" name="email" required>
                    </div>
                    
                    <div class="mb-3">
                        <label for="expertise" class="form-label">Areas of Expertise</label>
                        <input type="text" class="form-control" id="expertise" name="expertise" 
                               placeholder="e.g., machine learning, ethics, cybersecurity" required>
                        <div class="form-text">Separate multiple areas with commas</div>
                    </div>
                    
                    <button type="submit" class="btn btn-primary">Register</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        """
        
        # Dashboard template  
        dashboard_template = """
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-12">
        <h2>Evaluator Dashboard - {{ evaluator.name }}</h2>
        
        <div class="row mb-4">
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>{{ evaluator.completed_tasks }}</h5>
                        <p class="card-text">Completed Tasks</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>{{ assigned_tasks|length }}</h5>
                        <p class="card-text">Assigned Tasks</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>{{ "%.1f"|format(evaluator.reliability_score * 100) }}%</h5>
                        <p class="card-text">Reliability Score</p>
                    </div>
                </div>
            </div>
            
            <div class="col-md-3">
                <div class="card text-center">
                    <div class="card-body">
                        <h5>{{ "%.1f"|format(evaluator.average_time_per_task / 60) }}</h5>
                        <p class="card-text">Avg Time (min)</p>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <h4>Assigned Tasks</h4>
                {% for task in assigned_tasks %}
                <div class="card mb-2">
                    <div class="card-body">
                        <h6>{{ task.scenario_name }} - {{ task.model_name }}</h6>
                        <p class="card-text">{{ task.prompt[:100] }}...</p>
                        <a href="/evaluate/{{ task.task_id }}?evaluator_id={{ evaluator.evaluator_id }}" 
                           class="btn btn-primary btn-sm">Evaluate</a>
                    </div>
                </div>
                {% endfor %}
            </div>
            
            <div class="col-md-6">
                <h4>Available Tasks</h4>
                {% for task in pending_tasks %}
                <div class="card mb-2">
                    <div class="card-body">
                        <h6>{{ task.scenario_name }} - {{ task.model_name }}</h6>
                        <p class="card-text">{{ task.prompt[:100] }}...</p>
                        <button onclick="assignTask('{{ task.task_id }}', '{{ evaluator.evaluator_id }}')" 
                                class="btn btn-outline-primary btn-sm">Assign to Me</button>
                    </div>
                </div>
                {% endfor %}
            </div>
        </div>
    </div>
</div>
{% endblock %}
        """
        
        # Evaluation template
        evaluate_template = """
{% extends "base.html" %}

{% block content %}
<div class="row">
    <div class="col-md-8">
        <div class="card">
            <div class="card-header">
                <h4>Evaluate Response</h4>
                <small>{{ task.scenario_name }} - {{ task.model_name }}</small>
            </div>
            <div class="card-body">
                <div class="mb-3">
                    <h6>Original Prompt:</h6>
                    <div class="alert alert-light">{{ task.prompt }}</div>
                </div>
                
                <div class="mb-3">
                    <h6>Model Response:</h6>
                    <div class="alert alert-light">{{ task.model_response }}</div>
                </div>
                
                {% if task.attack_type %}
                <div class="mb-3">
                    <h6>Attack Type:</h6>
                    <span class="badge bg-warning">{{ task.attack_type }}</span>
                </div>
                {% endif %}
            </div>
        </div>
    </div>
    
    <div class="col-md-4">
        <div class="card">
            <div class="card-header">
                <h5>Evaluation Form</h5>
            </div>
            <div class="card-body">
                <form id="evaluationForm">
                    <input type="hidden" name="task_id" value="{{ task.task_id }}">
                    <input type="hidden" name="evaluator_id" value="{{ evaluator_id }}">
                    
                    {% for criterion_id, criterion in evaluation_criteria.items() %}
                    <div class="mb-3">
                        <label class="form-label">{{ criterion.name }}</label>
                        <small class="form-text text-muted">{{ criterion.description }}</small>
                        
                        <div class="mt-2">
                            {% for i in range(criterion.scale|length) %}
                            <div class="form-check">
                                <input class="form-check-input" type="radio" 
                                       name="{{ criterion_id }}" value="{{ criterion.scale[i] }}" 
                                       id="{{ criterion_id }}_{{ i }}" required>
                                <label class="form-check-label" for="{{ criterion_id }}_{{ i }}">
                                    {{ criterion.scale[i] }} - {{ criterion.labels[i] }}
                                </label>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                    {% endfor %}
                    
                    <div class="mb-3">
                        <label for="comments" class="form-label">Additional Comments</label>
                        <textarea class="form-control" name="comments" id="comments" rows="3"></textarea>
                    </div>
                    
                    <button type="submit" class="btn btn-success">Submit Evaluation</button>
                </form>
            </div>
        </div>
    </div>
</div>
{% endblock %}
        """
        
        # Write templates to files
        templates = {
            "base.html": base_template,
            "home.html": home_template,
            "register.html": register_template,
            "dashboard.html": dashboard_template,
            "evaluate.html": evaluate_template
        }
        
        for template_name, template_content in templates.items():
            template_path = self.templates_dir / template_name
            with open(template_path, "w") as f:
                f.write(template_content.strip())
    
    def _create_default_static_files(self):
        """Create default static files (CSS and JS)."""
        
        # CSS file
        css_content = """
.card {
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    border: none;
}

.navbar-brand {
    font-weight: bold;
}

.alert-light {
    background-color: #f8f9fa;
    border: 1px solid #dee2e6;
    max-height: 200px;
    overflow-y: auto;
}

.form-check {
    margin-bottom: 0.5rem;
}

.badge {
    font-size: 0.8em;
}

.btn-block {
    width: 100%;
    margin-bottom: 0.5rem;
}

.text-center h5 {
    font-size: 2rem;
    font-weight: bold;
    color: #007bff;
}
        """
        
        # JavaScript file
        js_content = """
// Registration form handler
document.addEventListener('DOMContentLoaded', function() {
    const registrationForm = document.getElementById('registrationForm');
    if (registrationForm) {
        registrationForm.addEventListener('submit', handleRegistration);
    }
    
    const evaluationForm = document.getElementById('evaluationForm');
    if (evaluationForm) {
        evaluationForm.addEventListener('submit', handleEvaluation);
    }
});

async function handleRegistration(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    
    try {
        const response = await fetch('/register', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(result.message);
            window.location.href = `/dashboard/${result.evaluator_id}`;
        } else {
            alert('Registration failed: ' + result.message);
        }
    } catch (error) {
        alert('Registration failed: ' + error.message);
    }
}

async function handleEvaluation(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    
    try {
        const response = await fetch('/submit_evaluation', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert(result.message);
            window.history.back();
        } else {
            alert('Evaluation submission failed: ' + result.message);
        }
    } catch (error) {
        alert('Evaluation submission failed: ' + error.message);
    }
}

async function assignTask(taskId, evaluatorId) {
    try {
        const response = await fetch('/api/assign_task', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                evaluator_id: evaluatorId,
                task_id: taskId
            })
        });
        
        const result = await response.json();
        
        if (result.success) {
            alert('Task assigned successfully!');
            location.reload();
        } else {
            alert('Task assignment failed: ' + result.message);
        }
    } catch (error) {
        alert('Task assignment failed: ' + error.message);
    }
}
        """
        
        # Write static files
        css_path = self.static_dir / "custom.css"
        with open(css_path, "w") as f:
            f.write(css_content.strip())
        
        js_path = self.static_dir / "app.js"
        with open(js_path, "w") as f:
            f.write(js_content.strip())
    
    def run(self, host: str = "0.0.0.0", port: int = 8080, debug: bool = False):
        """Run the web interface."""
        self.logger.info(f"Starting web interface on http://{host}:{port}")
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_level="info" if debug else "warning"
        )


# Standalone server function
def start_evaluation_server(evaluation_manager: HumanEvaluationManager, 
                           host: str = "0.0.0.0", port: int = 8080):
    """Start the evaluation web server."""
    interface = WebEvaluationInterface(evaluation_manager)
    interface.run(host=host, port=port, debug=True)
