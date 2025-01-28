from aryaxai.client.client import APIClient

def build_url(base_url, data_connector_name, project_name, organization_id):
    url = None
    if project_name:
        url = f"{base_url}?project_name={project_name}&link_service_name={data_connector_name}"
    elif organization_id:
        url = f"{base_url}?organization_id={organization_id}&link_service_name={data_connector_name}"
    return url

def build_list_data_connector_url(base_url, project_name, organization_id):
    url = None
    if project_name and organization_id:
        url = f"{base_url}?project_name={project_name}&organization_id={organization_id}"
    elif project_name:
        url = f"{base_url}?project_name={project_name}"
    elif organization_id:
        url = f"{base_url}?organization_id={organization_id}"
    else:
        url = base_url
    return url