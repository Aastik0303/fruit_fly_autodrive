const BASE = '/api'

async function request(path) {
  const response = await fetch(`${BASE}${path}`)
  if (!response.ok) {
    const body = await response.json().catch(() => ({}))
    throw new Error(body.detail || `${response.status} ${response.statusText}`)
  }
  return response.json()
}

// Neuron IDs are 18-digit integers, larger than a JavaScript number can hold exactly,
// so the API sends them as strings and they stay strings in the frontend.
export const api = {
  stats: () => request('/stats'),
  pointCloud: () => request('/visualization/neurons'),
  search: (query, limit = 20) => request(`/neurons/search?q=${encodeURIComponent(query)}&limit=${limit}`),
  neuron: (id) => request(`/neurons/${id}`),
  connections: (id, minSynapses = 1, limit = 50) =>
    request(`/neurons/${id}/connections?min_synapses=${minSynapses}&limit=${limit}`),
}
