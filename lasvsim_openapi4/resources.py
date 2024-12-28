"""
Resource module for the lasvsim API.
"""
from typing import Optional
from dataclasses import dataclass

from lasvsim_openapi4.http_client import HttpClient
from lasvsim_openapi4.qxmap import Qxmap


@dataclass
class GetHdMapReq:
    """Request for getting HD map."""
    scen_id: str = ""
    scen_ver: str = ""


@dataclass
class GetHdMapRes:
    """Response for getting HD map."""
    data: Optional[Qxmap] = None

    def __init__(self, data: dict = None):
        """Initialize response object.
        
        Args:
            data: Response data
        """
        if data is None:
            return
        self.data = Qxmap(**data["data"]) if data.get("data") else None


class Resources:
    """Resources client for the API."""
    http_client: HttpClient = None

    def __init__(self, http_client: HttpClient):
        """Initialize resources client.
        
        Args:
            http_client: HTTP client instance
        """
        self.http_client = http_client.clone()

    def get_hd_map(self, scen_id: str, scen_ver: str) -> GetHdMapRes:
        """Get HD map for a scenario.
        
        Args:
            scen_id: Scenario ID
            scen_ver: Scenario version
            
        Returns:
            HD map response
            
        Raises:
            APIError: If the request fails
        """
        req = GetHdMapReq(scen_id=scen_id, scen_ver=scen_ver)
        reply = GetHdMapRes()
        
        self.http_client.post(
            "/openapi/resource/v2/scenario/map/get",
            {"scen_id": req.scen_id, "scen_ver": req.scen_ver},
            reply
        )
        
        return reply
