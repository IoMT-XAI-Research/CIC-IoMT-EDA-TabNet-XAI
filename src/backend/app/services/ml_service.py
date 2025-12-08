import random
from .. import schemas

def get_mock_analysis(device_id: int) -> schemas.AnalysisResponse:
    # Simulate random risk score
    risk_score = random.uniform(0.0, 1.0)
    status = "ATTACK" if risk_score > 0.7 else "SAFE"
    
    return schemas.AnalysisResponse(
        device_status=status,
        risk_score=round(risk_score, 3),
        summary_text="Abnormal traffic pattern detected." if status == "ATTACK" else "Device operating normally.",
        xai_force_plot=schemas.XAIForcePlot(
            base_value=0.5,
            final_value=round(risk_score, 3),
            features=[
                {"name": "Outbound Packet Rate", "value": 0.3, "direction": "red"},
                {"name": "Connection Time", "value": 0.15, "direction": "red"},
                {"name": "Normal Traffic", "value": -0.02, "direction": "blue"}
            ]
        ),
        feature_importance_list=[
            {"name": "Outbound Packet Rate", "percentage": 70, "value_desc": "200% Increase"},
            {"name": "Connection Time", "percentage": 25, "value_desc": "High Latency"}
        ]
    )
