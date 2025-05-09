'use client';

import { useEffect, useState } from 'react';

interface Agent {
  agent_id: string;
  agent_type: string;
  model: string;
  tools: string[];
  prompt_name?: string;
  prompt_template?: string;
  has_memory: boolean;
  description?: string;
  metadata?: Record<string, any>;
}

export default function AgentsPage() {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchAgents() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch('http://localhost:8000/agents');
        if (!response.ok) {
          throw new Error(`Failed to fetch agents: ${response.status} ${response.statusText}`);
        }
        const data = await response.json();
        setAgents(data);
      } catch (err: any) {
        setError(err.message);
        console.error("Error fetching agents:", err);
      } finally {
        setLoading(false);
      }
    }

    fetchAgents();
  }, []);

  if (loading) {
    return <div className="container mx-auto p-4"><p className="text-center">Loading agents...</p></div>;
  }

  if (error) {
    return <div className="container mx-auto p-4"><p className="text-center text-red-500">Error: {error}</p></div>;
  }

  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Available AI Agents</h1>
      {agents.length === 0 ? (
        <p>No agents found.</p>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {agents.map((agent) => (
            <div key={agent.agent_id} className="border p-4 rounded-lg shadow">
              <h2 className="text-xl font-semibold">{agent.agent_id}</h2>
              <p className="text-sm text-gray-600">Type: {agent.agent_type}</p>
              <p className="text-sm text-gray-600">Model: {agent.model}</p>
              {agent.description && <p className="mt-2">{agent.description}</p>}
              {/* We can add buttons for 'Run Agent' or 'View Details' later */}
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 