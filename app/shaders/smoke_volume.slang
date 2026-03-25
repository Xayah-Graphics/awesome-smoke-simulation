struct PushConstants
{
    float4 eye;
    float4 right;
    float4 up;
    float4 forward;
    float4 volumeMin;
    float4 volumeMax;
    float4 colorA;
    float4 colorB;
    float4 params0;
    uint4 params1;
    uint4 params2;
    float4 params3;
};

struct VSOut
{
    float4 position : SV_Position;
    float2 ndc : TEXCOORD0;
};

[[vk::push_constant]]
ConstantBuffer<PushConstants> pc;

[[vk::binding(0, 0)]]
StructuredBuffer<float> fieldData;

static const uint RENDER_MODE_SMOKE = 0;
static const uint RENDER_MODE_SCALAR = 1;

uint field_index(uint x, uint y, uint z)
{
    return (z * pc.params1.y + y) * pc.params1.x + x;
}

float sample_field(float3 uvw)
{
    uvw = clamp(uvw, 0.0.xxx, 1.0.xxx);

    float3 p = uvw * float3(pc.params1.xyz - 1);
    uint3 lo = uint3(floor(p));
    uint3 hi = min(lo + 1, pc.params1.xyz - 1);
    float3 t = frac(p);

    float c000 = fieldData[field_index(lo.x, lo.y, lo.z)];
    float c100 = fieldData[field_index(hi.x, lo.y, lo.z)];
    float c010 = fieldData[field_index(lo.x, hi.y, lo.z)];
    float c110 = fieldData[field_index(hi.x, hi.y, lo.z)];
    float c001 = fieldData[field_index(lo.x, lo.y, hi.z)];
    float c101 = fieldData[field_index(hi.x, lo.y, hi.z)];
    float c011 = fieldData[field_index(lo.x, hi.y, hi.z)];
    float c111 = fieldData[field_index(hi.x, hi.y, hi.z)];

    float c00 = lerp(c000, c100, t.x);
    float c10 = lerp(c010, c110, t.x);
    float c01 = lerp(c001, c101, t.x);
    float c11 = lerp(c011, c111, t.x);
    float c0 = lerp(c00, c10, t.y);
    float c1 = lerp(c01, c11, t.y);
    return lerp(c0, c1, t.z);
}

float2 intersect_aabb(float3 rayOrigin, float3 rayDir, float3 boundsMin, float3 boundsMax)
{
    float3 safeDir = float3(
        abs(rayDir.x) < 1.0e-6 ? (rayDir.x < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.x,
        abs(rayDir.y) < 1.0e-6 ? (rayDir.y < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.y,
        abs(rayDir.z) < 1.0e-6 ? (rayDir.z < 0.0 ? -1.0e-6 : 1.0e-6) : rayDir.z);
    float3 invDir = 1.0.xxx / safeDir;
    float3 t0 = (boundsMin - rayOrigin) * invDir;
    float3 t1 = (boundsMax - rayOrigin) * invDir;
    float3 tMin = min(t0, t1);
    float3 tMax = max(t0, t1);

    float entry = max(max(tMin.x, tMin.y), tMin.z);
    float exit = min(min(tMax.x, tMax.y), tMax.z);
    return float2(entry, exit);
}

float3 render_smoke(float3 rayOrigin, float3 rayDir, float t0, float t1, float3 background)
{
    float distanceInside = t1 - t0;
    uint steps = max(pc.params1.w, 1u);
    float stepLength = distanceInside / steps;
    float transmittance = 1.0;
    float3 color = background;
    float softness = saturate(pc.params3.w);
    float densityGamma = lerp(1.3, 0.75, softness);

    [loop]
    for (uint i = 0; i < steps; ++i)
    {
        float travel = t0 + (i + 0.5) * stepLength;
        float3 worldPos = rayOrigin + travel * rayDir;
        float3 uvw = (worldPos - pc.volumeMin.xyz) / max(pc.volumeMax.xyz - pc.volumeMin.xyz, 1.0e-5.xxx);
        float density = max(sample_field(uvw) * pc.params0.z, 0.0);
        if (density < 1.0e-4)
        {
            continue;
        }

        density = pow(density, densityGamma);
        float leftToRight = smoothstep(0.12, 0.88, uvw.x);
        float heightTint = smoothstep(0.0, 0.45, uvw.y);
        float3 sampleColor = lerp(pc.colorA.rgb, pc.colorB.rgb, leftToRight);
        sampleColor *= lerp(0.78, 1.08, heightTint);

        float alpha = 1.0 - exp(-density * pc.params0.w * stepLength);
        color += transmittance * alpha * sampleColor;
        transmittance *= 1.0 - alpha;

        if (transmittance < 0.01)
        {
            break;
        }
    }

    return color;
}

float3 render_scalar(float3 rayOrigin, float3 rayDir, float t0, float t1, float3 background)
{
    float distanceInside = t1 - t0;
    uint steps = max(pc.params1.w, 1u);
    float stepLength = distanceInside / steps;
    float transmittance = 1.0;
    float3 color = background;
    float valueMin = pc.params3.x;
    float valueMax = max(pc.params3.y, valueMin + 1.0e-5);

    [loop]
    for (uint i = 0; i < steps; ++i)
    {
        float travel = t0 + (i + 0.5) * stepLength;
        float3 worldPos = rayOrigin + travel * rayDir;
        float3 uvw = (worldPos - pc.volumeMin.xyz) / max(pc.volumeMax.xyz - pc.volumeMin.xyz, 1.0e-5.xxx);
        float value = sample_field(uvw) * pc.params0.z;
        float normalized = saturate((value - valueMin) / (valueMax - valueMin));
        if (normalized < 1.0e-4)
        {
            continue;
        }

        float alpha = 1.0 - exp(-normalized * pc.params0.w * stepLength);
        float3 sampleColor = lerp(pc.colorA.rgb, pc.colorB.rgb, normalized);
        color += transmittance * alpha * sampleColor;
        transmittance *= 1.0 - alpha;

        if (transmittance < 0.01)
        {
            break;
        }
    }

    return color;
}

[shader("vertex")]
VSOut vs_main(uint vertexId : SV_VertexID)
{
    float2 positions[3] =
    {
        float2(-1.0, -1.0),
        float2(-1.0, 3.0),
        float2(3.0, -1.0),
    };

    VSOut output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    output.ndc = positions[vertexId];
    return output;
}

[shader("fragment")]
float4 fs_main(VSOut input) : SV_Target
{
    float2 uv = float2(input.ndc.x, -input.ndc.y);
    float3 rayDir = normalize(
        pc.forward.xyz +
        uv.x * pc.right.xyz * pc.params0.x * pc.params0.y +
        uv.y * pc.up.xyz * pc.params0.y);

    float3 rayOrigin = pc.eye.xyz;
    float2 hit = intersect_aabb(rayOrigin, rayDir, pc.volumeMin.xyz, pc.volumeMax.xyz);
    float t0 = max(hit.x, 0.0);
    float t1 = hit.y;

    float3 skyTop = float3(0.10, 0.12, 0.16);
    float3 skyBottom = float3(0.035, 0.04, 0.05);
    float horizon = saturate(0.5 + 0.5 * rayDir.y);
    float3 background = lerp(skyBottom, skyTop, horizon);

    if (t1 <= t0)
    {
        return float4(background, 1.0);
    }

    float3 color = pc.params2.x == RENDER_MODE_SMOKE
        ? render_smoke(rayOrigin, rayDir, t0, t1, background)
        : render_scalar(rayOrigin, rayDir, t0, t1, background);

    return float4(color, 1.0);
}
